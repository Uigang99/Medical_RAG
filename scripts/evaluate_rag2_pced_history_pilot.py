#!/usr/bin/env python3
"""Bounded, batched Rationale+Answer pilot for history-conditioned PCED.

The evaluator always regenerates a rerank-prior PCED control with the same
batched engine used by the proposed method.  It compares that control with the
legacy evaluator.  If the outputs match exactly, existing No-RAG, Base-RAG and
semantic-prior PCED outputs are reused.  Otherwise those baselines are
regenerated with this engine before aggregation.

The proposed condition keeps the existing PCED token score and rerank prior.
After each generated rationale token, every document expert receives a bounded
history update based on the selected token's log-probability relative to the
No-RAG stream.  That history changes only the expert prior at the next token.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_rag2_pced_direct_choice import (  # noqa: E402
    DATASETS,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    condition_summary,
    iter_jsonl,
    minmax,
    model_identity,
    paired_comparison,
    sha256_file,
)
from evaluate_rag2_pced_rationale_answer import (  # noqa: E402
    RationaleAnswerPcedGenerator,
    format_duration,
    load_inputs,
    load_no_rag,
    load_semantic,
    pair_key,
)
from medrag.core import BenchmarkSample  # noqa: E402
from medrag.rag2_anchored_trace import (  # noqa: E402
    CHOICES,
    canonical_response,
    normalize_rationale,
)


RUN_VERSION = "rag2_pced_history_rationale_answer_batched_pilot_v1"
HISTORY_RULE = "bounded_ema_selected_token_logprob_ratio_to_no_rag_v1"
DEFAULT_LLAMA = WORKSPACE_ROOT / "models/Llama-3-8B-Instruct"
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2/top8/candidates.jsonl"
)
DEFAULT_SEMANTIC = (
    PROJECT_ROOT
    / "results/rag2_pced_topk_answer_mode_sweep_three_anchor_v2/direct_choice/top8/"
    "semantic_support_probabilities.jsonl"
)
DEFAULT_LEGACY = (
    PROJECT_ROOT
    / "results/rag2_pced_topk_answer_mode_sweep_three_anchor_v2/rationale_answer/top8"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/rag2_pced_history_rationale_answer_top8_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--semantic-score-cache", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--legacy-output-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-root", type=Path, default=PROJECT_ROOT / "datasets/benchmark")
    parser.add_argument("--collection", default="unified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--no-rag-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--questions-per-dataset", type=int, default=64)
    parser.add_argument("--fidelity-per-dataset", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--prior-epsilon", type=float, default=1e-4)
    parser.add_argument("--history-strength", type=float, default=0.10)
    parser.add_argument("--history-decay", type=float, default=0.95)
    parser.add_argument("--history-temperature", type=float, default=1.0)
    parser.add_argument("--history-cap", type=float, default=2.0)
    parser.add_argument("--pced-question-batch-size", type=int, default=32)
    parser.add_argument("--plain-question-batch-size", type=int, default=64)
    parser.add_argument("--choice-prompt-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-rationale-tokens", type=int, default=512)
    parser.add_argument("--answer-reserve-tokens", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def select_balanced(
    samples: Sequence[BenchmarkSample],
    candidates: Sequence[dict[str, Any]],
    per_dataset: int,
    seed: int,
) -> tuple[list[BenchmarkSample], list[dict[str, Any]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample.dataset].append(index)
    missing = {
        name: per_dataset - len(grouped[name])
        for name in DATASETS
        if len(grouped[name]) < per_dataset
    }
    if missing:
        raise RuntimeError(f"Insufficient balanced pilot rows: {missing}")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for name in DATASETS:
        chosen.extend(int(value) for value in rng.choice(grouped[name], size=per_dataset, replace=False))
    # Restore canonical benchmark order after seeded stratified selection so
    # batching and legacy joins do not depend on random permutation order.
    chosen.sort()
    return [samples[index] for index in chosen], [candidates[index] for index in chosen]


def selected_fidelity_ids(samples: Sequence[BenchmarkSample], per_dataset: int) -> set[str]:
    counts: Counter[str] = Counter()
    result: set[str] = set()
    for sample in samples:
        if counts[sample.dataset] < per_dataset:
            counts[sample.dataset] += 1
            result.add(sample.id)
    return result


def legacy_condition(path: Path, condition: str, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    root = path / "generation_shards" / condition
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, dict[str, Any]] = {}
    for shard in sorted(root.glob("shard_*.jsonl")):
        for row in iter_jsonl(shard):
            identifier = str(row.get("sample_id") or "")
            if identifier in selected_ids:
                result[identifier] = row
    missing = selected_ids - set(result)
    if missing:
        raise RuntimeError(f"Legacy {condition} output misses {len(missing)} selected rows; first={next(iter(missing))}")
    return result


def matched_stop_length(generator: RationaleAnswerPcedGenerator, generated: Sequence[int]) -> int:
    if generated and generator.tokenizer.eos_token_id is not None and generated[-1] == generator.tokenizer.eos_token_id:
        return 1
    matches = [
        len(stop)
        for stop in generator.stop_sequences
        if stop and len(generated) >= len(stop) and list(generated[-len(stop) :]) == stop
    ]
    return max(matches, default=0)


def update_history(
    history: torch.Tensor,
    advantage: torch.Tensor,
    *,
    decay: float,
    temperature: float,
    cap: float,
) -> torch.Tensor:
    """Apply a bounded signed update around the No-RAG log-probability boundary."""
    return (decay * history + torch.tanh(advantage / temperature)).clamp(-cap, cap)


def rapid_switchbacks(winners: Sequence[int], window: int = 3) -> int:
    """Count A->...->A returns that include another expert within `window` tokens."""
    count = 0
    for right in range(2, len(winners)):
        left_limit = max(0, right - window)
        for left in range(right - 2, left_limit - 1, -1):
            if winners[left] == winners[right] and any(
                winners[index] != winners[right] for index in range(left + 1, right)
            ):
                count += 1
                break
    return count


class PilotProgress:
    def __init__(self, stages: Sequence[str]) -> None:
        self.stages = list(stages)
        self.started = time.time()
        self.stage_started = self.started
        self.index = 0
        self.total = 1
        self.done = 0
        self.initial = 0
        self.future_question_equivalents: int | None = None
        self.unit = "question"
        self.bar: tqdm[Any] | None = None

    def start(
        self,
        index: int,
        total: int,
        *,
        initial: int = 0,
        future_question_equivalents: int | None = None,
        detail: str = "",
        unit: str = "question",
    ) -> None:
        if self.bar is not None:
            self.bar.close()
        self.index = index
        self.total = max(1, int(total))
        self.done = int(initial)
        self.initial = int(initial)
        self.future_question_equivalents = (
            None if future_question_equivalents is None else int(future_question_equivalents)
        )
        self.unit = unit
        self.stage_started = time.time()
        stage_label = self.stages[index - 1] + (f" [{detail}]" if detail else "")
        print(
            f"[overall {index}/{len(self.stages)} | elapsed {format_duration(time.time()-self.started)} | "
            f"overall ETA unknown until this stage measures throughput] {stage_label}",
            flush=True,
        )
        self.bar = tqdm(
            total=total,
            initial=initial,
            unit=unit,
            dynamic_ncols=True,
            desc=f"Stage {index}/{len(self.stages)} - {stage_label}",
        )

    def update(self, amount: int) -> None:
        self.done += int(amount)
        if self.bar is not None:
            self.bar.update(int(amount))
            elapsed = max(time.time() - self.stage_started, 1e-6)
            active = max(0, self.done - self.initial)
            rate = active / elapsed
            stage_eta = (self.total - self.done) / rate if rate > 0 else None
            overall_eta = None
            if stage_eta is not None and self.future_question_equivalents is not None:
                overall_eta = stage_eta + self.future_question_equivalents / max(rate, 1e-12)
            stage_pct = 100.0 * self.done / self.total
            overall_pct = 100.0 * ((self.index - 1) + self.done / self.total) / len(self.stages)
            self.bar.set_postfix_str(
                f"overall={overall_pct:.1f}% stage={stage_pct:.1f}% rate={rate:.2f}{self.unit}/s "
                f"stage_ETA={format_duration(stage_eta)} overall_ETA={format_duration(overall_eta)}",
                refresh=False,
            )

    def complete(self, detail: str) -> None:
        if self.done < self.total:
            self.update(self.total - self.done)
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        print(
            f"[stage {self.index}/{len(self.stages)} complete | duration "
            f"{format_duration(time.time()-self.stage_started)}] {detail}",
            flush=True,
        )


class BatchedGenerator(RationaleAnswerPcedGenerator):
    @torch.inference_mode()
    def _advance_many(
        self, question_tokens: torch.Tensor, streams: int, cache: Any, mask: torch.Tensor
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        current = question_tokens.repeat_interleave(streams).unsqueeze(1).to(self.device)
        next_mask = torch.cat([mask, torch.ones_like(current)], dim=1)
        output = self.model(
            input_ids=current,
            attention_mask=next_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return output.logits[:, -1].float(), output.past_key_values, next_mask

    @staticmethod
    def jsd_beta_batch(logits: torch.Tensor) -> torch.Tensor:
        # logits: [questions, no-rag + documents, vocabulary]
        amateur_log = F.log_softmax(logits[:, 0], dim=-1)
        expert_log = F.log_softmax(logits[:, 1:], dim=-1)
        amateur = amateur_log.exp().unsqueeze(1).expand_as(expert_log)
        experts = expert_log.exp()
        mixture = 0.5 * (amateur + experts)
        log_mixture = mixture.clamp_min(1e-12).log()
        return 0.5 * (
            (experts * (expert_log - log_mixture)).sum(-1)
            + (amateur * (amateur_log.unsqueeze(1) - log_mixture)).sum(-1)
        ).mean(-1)

    @torch.inference_mode()
    def _choice_logits_many(
        self,
        samples: Sequence[BenchmarkSample],
        document_texts: Sequence[Sequence[str | None]],
        rationales: Sequence[str],
    ) -> torch.Tensor:
        prompts: list[str] = []
        streams = len(document_texts[0])
        for sample, texts, rationale in zip(samples, document_texts, rationales, strict=True):
            if len(texts) != streams:
                raise RuntimeError("Choice stream count changed inside a batch")
            prompts.extend(
                self._choice_prompt(sample, text, rationale)
                for text in texts
            )
        chunks: list[torch.Tensor] = []
        for start in range(0, len(prompts), self.args.choice_prompt_batch_size):
            logits, cache, mask = self._prefill(prompts[start : start + self.args.choice_prompt_batch_size])
            chunks.append(logits.index_select(-1, self.choice_ids).cpu())
            del cache, mask, logits
        return torch.cat(chunks, dim=0).reshape(len(samples), streams, len(CHOICES))

    def _choice_prompt(self, sample: BenchmarkSample, document_text: str | None, rationale: str) -> str:
        # Reuse the exact established implementation instead of reimplementing
        # the anchored prompt contract here.
        from medrag.rag2_anchored_trace import (
            assistant_decision_prefix,
            build_anchored_user_prompt,
            render_chat_prompt,
        )

        return (
            render_chat_prompt(self.tokenizer, build_anchored_user_prompt(sample.raw, document_text))
            + assistant_decision_prefix(rationale)
        )

    @torch.inference_mode()
    def generate_plain_batch(
        self,
        samples: Sequence[BenchmarkSample],
        candidates: Sequence[dict[str, Any]],
        condition: str,
    ) -> list[dict[str, Any]]:
        if condition not in {"no_rag", "base_rag"}:
            raise ValueError(condition)
        texts: list[str | None] = []
        packing: list[dict[str, Any] | None] = []
        for sample, row in zip(samples, candidates, strict=True):
            if condition == "no_rag":
                texts.append(None)
                packing.append(None)
            else:
                text, metadata = self.fit_documents(
                    sample, [str(document["text"]) for document in row["reranked_documents"]]
                )
                texts.append(text)
                packing.append(metadata)
        logits, cache, mask = self._prefill(
            [self._prompt(sample, text) for sample, text in zip(samples, texts, strict=True)]
        )
        batch = len(samples)
        generated: list[list[int]] = [[] for _ in samples]
        active = [True] * batch
        finish = ["length"] * batch
        eos = int(self.tokenizer.eos_token_id or 0)
        for _ in range(self.args.max_rationale_tokens):
            token_values = [int(value) for value in logits.argmax(-1).cpu().tolist()]
            for index in range(batch):
                if not active[index]:
                    token_values[index] = eos
                    continue
                token = token_values[index]
                generated[index].append(token)
                if self._stopped(generated[index]):
                    active[index] = False
                    finish[index] = "stop"
            if not any(active):
                break
            tokens = torch.tensor(token_values, dtype=torch.long, device=self.device)
            logits, cache, mask = self._advance_many(tokens, 1, cache, mask)
        raw = [
            self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for tokens in generated
        ]
        normalized = [normalize_rationale(text) for text in raw]
        rationales = [item[0] for item in normalized]
        choice = self._choice_logits_many(
            samples, [[text] for text in texts], rationales
        )[:, 0]
        results: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            answer = CHOICES[int(choice[index].argmax().item())]
            flags = list(normalized[index][1])
            if finish[index] == "length":
                flags.append("rationale_length_exhausted")
            results.append({
                "sample_id": sample.id,
                "dataset": sample.dataset,
                "gold_answer": sample.answer,
                "answer": answer,
                "correct": answer == sample.answer,
                "rationale": rationales[index],
                "canonical_response": canonical_response(rationales[index], answer, sample.options or {}),
                "rationale_tokens": len(generated[index]),
                "finish_reason": finish[index],
                "quality_flags": sorted(set(flags)),
                "choice_logits": [float(value) for value in choice[index].tolist()],
                "prompt_packing": packing[index],
            })
        del cache, mask, logits
        return results

    @torch.inference_mode()
    def generate_pced_batch(
        self,
        samples: Sequence[BenchmarkSample],
        candidates: Sequence[dict[str, Any]],
        priors: Sequence[np.ndarray],
        *,
        history_enabled: bool,
    ) -> list[dict[str, Any]]:
        batch = len(samples)
        streams = self.args.top_k + 1
        all_fitted: list[list[str]] = []
        all_packing: list[list[dict[str, Any]]] = []
        prompts: list[str] = []
        for sample, row in zip(samples, candidates, strict=True):
            fitted: list[str] = []
            packing: list[dict[str, Any]] = []
            for document in row["reranked_documents"]:
                text, metadata = self.fit_documents(sample, [str(document["text"])])
                fitted.append(text)
                packing.append(metadata)
            all_fitted.append(fitted)
            all_packing.append(packing)
            prompts.append(self._prompt(sample, None))
            prompts.extend(self._prompt(sample, text) for text in fitted)

        flat_logits, cache, mask = self._prefill(prompts)
        logits = flat_logits.reshape(batch, streams, -1)
        beta = self.jsd_beta_batch(logits)
        prior = torch.tensor(np.stack(priors), dtype=torch.float32, device=self.device).clamp_min(
            self.args.prior_epsilon
        )
        history = torch.zeros((batch, self.args.top_k), dtype=torch.float32, device=self.device)
        generated: list[list[int]] = [[] for _ in samples]
        winners: list[list[int]] = [[] for _ in samples]
        advantages: list[list[list[float]]] = [[] for _ in samples]
        active = [True] * batch
        finish = ["length"] * batch
        eos = int(self.tokenizer.eos_token_id or 0)

        for _ in range(self.args.max_rationale_tokens):
            scores = (
                (1.0 + beta[:, None, None]) * logits[:, 1:]
                - beta[:, None, None] * logits[:, 0, None]
                + self.args.gamma * prior.log().unsqueeze(-1)
            )
            if history_enabled:
                scores = scores + self.args.history_strength * history.unsqueeze(-1)
            flat = scores.reshape(batch, -1).argmax(-1)
            tokens = flat.remainder(scores.shape[-1])
            selected_experts = flat.div(scores.shape[-1], rounding_mode="floor")

            log_normalizer = torch.logsumexp(logits, dim=-1)
            chosen_logits = logits.gather(
                2, tokens[:, None, None].expand(-1, streams, 1)
            ).squeeze(-1)
            selected_log_probability = chosen_logits - log_normalizer
            advantage = selected_log_probability[:, 1:] - selected_log_probability[:, :1]
            # One device synchronization per decoding step.  Repeated scalar
            # .item() calls here make multi-question decoding much slower.
            step_values = torch.cat(
                [tokens[:, None].float(), selected_experts[:, None].float(), advantage], dim=1
            ).cpu().tolist()

            for index in range(batch):
                if not active[index]:
                    step_values[index][0] = eos
                    continue
                generated[index].append(int(step_values[index][0]))
                winners[index].append(int(step_values[index][1]))
                advantages[index].append([float(value) for value in step_values[index][2:]])
                if self._stopped(generated[index]):
                    active[index] = False
                    finish[index] = "stop"

            if history_enabled:
                updated = update_history(
                    history,
                    advantage,
                    decay=self.args.history_decay,
                    temperature=self.args.history_temperature,
                    cap=self.args.history_cap,
                )
                active_tensor = torch.tensor(active, dtype=torch.bool, device=self.device)
                history = torch.where(active_tensor[:, None], updated, history)
            if not any(active):
                break
            tokens = torch.tensor(
                [int(row[0]) for row in step_values], dtype=torch.long, device=self.device
            )
            flat_logits, cache, mask = self._advance_many(tokens, streams, cache, mask)
            logits = flat_logits.reshape(batch, streams, -1)

        raw = [
            self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for tokens in generated
        ]
        normalized = [normalize_rationale(text) for text in raw]
        rationales = [item[0] for item in normalized]

        # Remove stop-marker updates before the constrained final answer.  The
        # marker is formatting, not rationale evidence.
        content_winners: list[list[int]] = []
        final_history_rows: list[torch.Tensor] = []
        for index in range(batch):
            stop_length = matched_stop_length(self, generated[index])
            content_length = max(0, len(generated[index]) - stop_length)
            content_winners.append(winners[index][:content_length])
            value = torch.zeros(self.args.top_k, dtype=torch.float32, device=self.device)
            if history_enabled:
                for row in advantages[index][:content_length]:
                    tensor = torch.tensor(row, dtype=torch.float32, device=self.device)
                    value = update_history(
                        value,
                        tensor,
                        decay=self.args.history_decay,
                        temperature=self.args.history_temperature,
                        cap=self.args.history_cap,
                    )
            final_history_rows.append(value)
        final_history = torch.stack(final_history_rows)

        choice_logits = self._choice_logits_many(
            samples,
            [[None] + fitted for fitted in all_fitted],
            rationales,
        ).to(self.device)
        choice_scores = (
            (1.0 + beta[:, None, None]) * choice_logits[:, 1:]
            - beta[:, None, None] * choice_logits[:, :1]
            + self.args.gamma * prior.log().unsqueeze(-1)
        )
        if history_enabled:
            choice_scores = choice_scores + self.args.history_strength * final_history.unsqueeze(-1)
        flat_choice = choice_scores.reshape(batch, -1).argmax(-1)
        answer_index = flat_choice.remainder(len(CHOICES))
        final_expert = flat_choice.div(len(CHOICES), rounding_mode="floor")

        results: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            answer = CHOICES[int(answer_index[index].item())]
            counts = Counter(winners[index])
            content_counts = Counter(content_winners[index])
            content_advantage = [
                row[winner]
                for row, winner in zip(advantages[index][: len(content_winners[index])], content_winners[index], strict=True)
            ]
            flags = list(normalized[index][1])
            if finish[index] == "length":
                flags.append("rationale_length_exhausted")
            results.append({
                "sample_id": sample.id,
                "dataset": sample.dataset,
                "gold_answer": sample.answer,
                "answer": answer,
                "correct": answer == sample.answer,
                "rationale": rationales[index],
                "canonical_response": canonical_response(rationales[index], answer, sample.options or {}),
                "rationale_tokens": len(generated[index]),
                "rationale_content_tokens": len(content_winners[index]),
                "finish_reason": finish[index],
                "quality_flags": sorted(set(flags)),
                "beta": float(beta[index].item()),
                "prior": [float(value) for value in prior[index].cpu().tolist()],
                "history_enabled": history_enabled,
                "history_state_final": [float(value) for value in final_history[index].cpu().tolist()],
                "history_rule": HISTORY_RULE if history_enabled else "disabled_control",
                "rationale_expert_token_counts": {
                    str(rank + 1): int(count) for rank, count in counts.items()
                },
                "rationale_content_expert_token_counts": {
                    str(rank + 1): int(count) for rank, count in content_counts.items()
                },
                "rationale_expert_switches": sum(
                    left != right for left, right in zip(winners[index], winners[index][1:])
                ),
                "rationale_content_expert_switches": sum(
                    left != right
                    for left, right in zip(content_winners[index], content_winners[index][1:])
                ),
                "rationale_content_rapid_switchbacks": rapid_switchbacks(content_winners[index]),
                "rationale_content_winner_sequence": [rank + 1 for rank in content_winners[index]],
                "selected_expert_no_rag_logprob_advantage_mean": (
                    float(np.mean(content_advantage)) if content_advantage else None
                ),
                "final_expert_rank": int(final_expert[index].item()) + 1,
                "choice_score_by_option": [
                    float(value) for value in choice_scores[index].max(dim=0).values.cpu().tolist()
                ],
                "expert_prompt_packing": all_packing[index],
            })
        del cache, mask, flat_logits, logits
        return results


def valid_shard(path: Path, expected_ids: Sequence[str], contract_hash: str) -> bool:
    marker = path.with_suffix(".complete.json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        rows = list(iter_jsonl(path))
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        metadata.get("contract_hash") == contract_hash
        and [str(row.get("sample_id") or "") for row in rows] == list(expected_ids)
    )


def run_cached_condition(
    *,
    args: argparse.Namespace,
    condition: str,
    samples: Sequence[BenchmarkSample],
    candidates: Sequence[dict[str, Any]],
    semantic: dict[str, float],
    generator: BatchedGenerator,
    contract_hash: str,
    progress: PilotProgress,
    stage_index: int,
    future_question_equivalents: int | None,
) -> list[dict[str, Any]]:
    root = args.output_dir / "generation_shards" / condition
    root.mkdir(parents=True, exist_ok=True)
    slices = [
        (start, min(len(samples), start + args.shard_size))
        for start in range(0, len(samples), args.shard_size)
    ]
    completed = 0
    for shard_index, (start, stop) in enumerate(slices):
        path = root / f"shard_{shard_index:05d}.jsonl"
        if valid_shard(path, [sample.id for sample in samples[start:stop]], contract_hash):
            completed += stop - start
    progress.start(
        stage_index,
        len(samples),
        initial=completed,
        future_question_equivalents=future_question_equivalents,
        detail=condition,
    )

    for shard_index, (start, stop) in enumerate(slices):
        path = root / f"shard_{shard_index:05d}.jsonl"
        expected = [sample.id for sample in samples[start:stop]]
        if valid_shard(path, expected, contract_hash):
            continue
        output: list[dict[str, Any]] = []
        cursor = start
        requested_batch = (
            args.plain_question_batch_size if condition in {"no_rag", "base_rag"}
            else args.pced_question_batch_size
        )
        active_batch = min(requested_batch, stop - start)
        shard_started = time.time()
        while cursor < stop:
            size = min(active_batch, stop - cursor)
            batch_samples = samples[cursor : cursor + size]
            batch_candidates = candidates[cursor : cursor + size]
            try:
                if condition in {"no_rag", "base_rag"}:
                    rows = generator.generate_plain_batch(batch_samples, batch_candidates, condition)
                else:
                    priors: list[np.ndarray] = []
                    for sample, candidate in zip(batch_samples, batch_candidates, strict=True):
                        documents = candidate["reranked_documents"]
                        if condition in {"pced_rerank", "pced_history"}:
                            values = minmax(
                                [float(document["rerank_score"]) for document in documents],
                                args.prior_epsilon,
                            )
                        elif condition == "pced_semantic":
                            values = np.asarray(
                                [semantic[pair_key(sample, document)] for document in documents],
                                dtype=np.float64,
                            )
                            values = np.clip(values, args.prior_epsilon, 1.0 - args.prior_epsilon)
                        else:
                            raise ValueError(condition)
                        priors.append(values)
                    rows = generator.generate_pced_batch(
                        batch_samples,
                        batch_candidates,
                        priors,
                        history_enabled=condition == "pced_history",
                    )
            except torch.cuda.OutOfMemoryError:
                if active_batch <= 1:
                    raise
                active_batch = max(1, active_batch // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"[OOM recovery] condition={condition} cursor={cursor}/{stop} "
                    f"question_batch={active_batch}; retrying without losing a durable shard",
                    flush=True,
                )
                continue
            output.extend(rows)
            cursor += size
            progress.update(size)
        atomic_jsonl(path, output)
        atomic_json(path.with_suffix(".complete.json"), {
            "contract_hash": contract_hash,
            "condition": condition,
            "shard": shard_index,
            "questions": len(output),
            "effective_question_batch_size": active_batch,
            "elapsed_seconds": time.time() - shard_started,
        })
    progress.complete(f"condition={condition} cache={root}")
    result: list[dict[str, Any]] = []
    for shard_index, (start, stop) in enumerate(slices):
        path = root / f"shard_{shard_index:05d}.jsonl"
        if not valid_shard(path, [sample.id for sample in samples[start:stop]], contract_hash):
            raise RuntimeError(f"Incomplete shard after generation: {path}")
        result.extend(iter_jsonl(path))
    return result


def fidelity_report(
    samples: Sequence[BenchmarkSample],
    current: Sequence[dict[str, Any]],
    legacy: dict[str, dict[str, Any]],
    selected_ids: set[str],
) -> dict[str, Any]:
    indexed = {str(row["sample_id"]): row for row in current}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.id not in selected_ids:
            continue
        new = indexed[sample.id]
        old = legacy[sample.id]
        rows.append({
            "sample_id": sample.id,
            "dataset": sample.dataset,
            "answer_match": str(new["answer"]) == str(old["answer"]),
            "rationale_match": str(new["rationale"]) == str(old["rationale"]),
            "rationale_token_count_match": int(new["rationale_tokens"]) == int(old["rationale_tokens"]),
        })
    count = len(rows)
    report = {
        "questions": count,
        "answer_matches": sum(row["answer_match"] for row in rows),
        "rationale_exact_matches": sum(row["rationale_match"] for row in rows),
        "rationale_token_count_matches": sum(row["rationale_token_count_match"] for row in rows),
        "exact_pass": bool(rows) and all(
            row["answer_match"] and row["rationale_match"] and row["rationale_token_count_match"]
            for row in rows
        ),
        "rows": rows,
    }
    return report


def history_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    token_total = sum(int(row.get("rationale_content_tokens") or 0) for row in rows)
    switches = sum(int(row.get("rationale_content_expert_switches") or 0) for row in rows)
    returns = sum(int(row.get("rationale_content_rapid_switchbacks") or 0) for row in rows)
    runs: list[float] = []
    collapsed = 0
    advantages: list[float] = []
    for row in rows:
        tokens = int(row.get("rationale_content_tokens") or 0)
        counts = row.get("rationale_content_expert_token_counts") or {}
        local_switches = int(row.get("rationale_content_expert_switches") or 0)
        if tokens:
            runs.append(tokens / (local_switches + 1))
            collapsed += int(max(map(int, counts.values()), default=0) / tokens >= 0.9)
        value = row.get("selected_expert_no_rag_logprob_advantage_mean")
        if value is not None and math.isfinite(float(value)):
            advantages.append(float(value))
    return {
        "questions": len(rows),
        "content_tokens": token_total,
        "expert_switches_per_100_content_tokens": 100.0 * switches / max(1, token_total),
        "rapid_switchbacks_per_100_content_tokens": 100.0 * returns / max(1, token_total),
        "mean_expert_run_length_tokens": float(np.mean(runs)) if runs else None,
        "expert_90pct_collapse_rate": collapsed / max(1, len(rows)),
        "mean_selected_expert_no_rag_logprob_advantage": (
            float(np.mean(advantages)) if advantages else None
        ),
    }


def aggregate(
    args: argparse.Namespace,
    samples: Sequence[BenchmarkSample],
    conditions: dict[str, Sequence[dict[str, Any]]],
    fidelity: dict[str, Any],
    baseline_source: str,
    progress: PilotProgress,
) -> None:
    progress.start(8, 6, unit="task", future_question_equivalents=0)
    indexed = {
        name: {str(row["sample_id"]): row for row in rows}
        for name, rows in conditions.items()
    }
    results: list[dict[str, Any]] = []
    for sample in samples:
        predictions = {name: str(rows[sample.id]["answer"]).upper() for name, rows in indexed.items()}
        results.append({
            "sample_id": sample.id,
            "dataset": sample.dataset,
            "row_idx": sample.row_idx,
            "gold_answer": sample.answer,
            "predictions": predictions,
        })
    progress.update(1)
    metrics = {name: condition_summary(results, name) for name in conditions}
    specs = (
        ("Base-RAG vs No-RAG", "base_rag", "no_rag"),
        ("PCED rerank vs Base-RAG", "pced_rerank", "base_rag"),
        ("PCED semantic vs PCED rerank", "pced_semantic", "pced_rerank"),
        ("PCED history vs PCED rerank", "pced_history", "pced_rerank"),
        ("PCED history vs Base-RAG", "pced_history", "base_rag"),
    )
    comparisons: dict[str, Any] = {}
    for index, (label, condition, baseline) in enumerate(specs):
        comparisons[label] = paired_comparison(
            results, condition, baseline, args.bootstrap_replicates, args.seed + index
        )
        progress.update(1)
    routing = {
        "pced_rerank": history_diagnostics(conditions["pced_rerank"]),
        "pced_history": history_diagnostics(conditions["pced_history"]),
    }
    baseline_routing = routing["pced_rerank"]
    history_routing = routing["pced_history"]
    baseline_returns = baseline_routing["rapid_switchbacks_per_100_content_tokens"]
    rapid_return_reduction = (
        None
        if baseline_returns <= 0
        else 1.0 - history_routing["rapid_switchbacks_per_100_content_tokens"] / baseline_returns
    )
    accuracy_delta = comparisons["PCED history vs PCED rerank"]["accuracy_delta"]
    baseline_advantage = baseline_routing["mean_selected_expert_no_rag_logprob_advantage"]
    history_advantage = history_routing["mean_selected_expert_no_rag_logprob_advantage"]
    advantage_delta = (
        None
        if baseline_advantage is None or history_advantage is None
        else history_advantage - baseline_advantage
    )
    collapse_delta = (
        history_routing["expert_90pct_collapse_rate"]
        - baseline_routing["expert_90pct_collapse_rate"]
    )
    criteria = {
        "rapid_switchback_reduction_at_least_20pct": (
            rapid_return_reduction is not None and rapid_return_reduction >= 0.20
        ),
        "selected_expert_advantage_not_lower": (
            advantage_delta is not None and advantage_delta >= 0.0
        ),
        "one_expert_collapse_increase_at_most_5pp": collapse_delta <= 0.05,
        "micro_accuracy_drop_at_most_1pp": accuracy_delta >= -0.01,
    }
    decision = {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "rapid_switchback_relative_reduction": rapid_return_reduction,
        "selected_expert_advantage_delta": advantage_delta,
        "one_expert_collapse_rate_delta": collapse_delta,
        "micro_accuracy_delta": accuracy_delta,
    }
    summary = {
        "run_version": RUN_VERSION,
        "questions": len(samples),
        "questions_per_dataset": args.questions_per_dataset,
        "top_k": args.top_k,
        "baseline_source": baseline_source,
        "fidelity": {key: value for key, value in fidelity.items() if key != "rows"},
        "history_hyperparameters": {
            "strength": args.history_strength,
            "decay": args.history_decay,
            "temperature": args.history_temperature,
            "cap": args.history_cap,
        },
        "conditions": metrics,
        "paired_comparisons": comparisons,
        "routing_diagnostics": routing,
        "pilot_decision": decision,
    }
    atomic_jsonl(args.output_dir / "predictions.jsonl", results)
    atomic_json(args.output_dir / "summary.json", summary)
    labels = {
        "no_rag": "No-RAG",
        "base_rag": "Base-RAG",
        "pced_rerank": "PCED rerank",
        "pced_semantic": "PCED semantic",
        "pced_history": "PCED accumulated history",
    }
    lines = [
        f"Rationale+Answer PCED history pilot (Top-{args.top_k}, N={len(samples)})",
        f"Baseline source: {baseline_source}; legacy exact fidelity: {fidelity['exact_pass']}",
        "",
        "| Condition | N | MedMCQA | MedQA | MMLU pooled | Micro | Macro-8 | Macro-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in metrics.items():
        pct = lambda item: "—" if item is None else f"{100*item:.2f}"  # noqa: E731
        lines.append(
            f"| {labels[name]} | {value['questions']} | {pct(value['medmcqa_accuracy'])} | "
            f"{pct(value['medqa_accuracy'])} | {pct(value['mmlu_pooled_accuracy'])} | "
            f"{pct(value['micro_accuracy'])} | {pct(value['macro8_accuracy'])} | "
            f"{pct(value['macro3_accuracy'])} |"
        )
    lines.extend(["", "Routing consistency (rationale content tokens only):", ""])
    lines.append("| Condition | switches / 100 tokens | rapid returns / 100 tokens | mean run | >=90% one-expert | selected-expert advantage |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("pced_rerank", "pced_history"):
        value = routing[name]
        lines.append(
            f"| {labels[name]} | {value['expert_switches_per_100_content_tokens']:.3f} | "
            f"{value['rapid_switchbacks_per_100_content_tokens']:.3f} | "
            f"{value['mean_expert_run_length_tokens']:.3f} | "
            f"{100*value['expert_90pct_collapse_rate']:.2f}% | "
            f"{value['mean_selected_expert_no_rag_logprob_advantage']:.5f} |"
        )
    lines.extend(["", "Paired accuracy changes:", ""])
    for label, value in comparisons.items():
        interval = value["paired_bootstrap_95ci"]
        lines.append(
            f"- {label}: {100*value['accuracy_delta']:+.2f}%p "
            f"(95% CI {100*interval[0]:+.2f} to {100*interval[1]:+.2f}; "
            f"W->C={value['wrong_to_correct']}, C->W={value['correct_to_wrong']})"
        )
    lines.extend(["", f"Predefined pilot decision: {'PASS' if decision['passed'] else 'FAIL'}", ""])
    lines.append(
        f"- rapid switchback reduction >=20%: {criteria['rapid_switchback_reduction_at_least_20pct']} "
        f"(observed={rapid_return_reduction})"
    )
    lines.append(
        f"- selected-expert No-RAG advantage not lower: {criteria['selected_expert_advantage_not_lower']} "
        f"(delta={'undefined' if advantage_delta is None else f'{advantage_delta:+.6f}'})"
    )
    lines.append(
        f"- >=90% one-expert collapse increase <=5%p: "
        f"{criteria['one_expert_collapse_increase_at_most_5pp']} (delta={100*collapse_delta:+.2f}%p)"
    )
    lines.append(
        f"- micro accuracy drop <=1%p: {criteria['micro_accuracy_drop_at_most_1pp']} "
        f"(delta={100*accuracy_delta:+.2f}%p)"
    )
    table = "\n".join(lines) + "\n"
    (args.output_dir / "summary_table_pretty.txt").write_text(table, encoding="utf-8")
    progress.complete(
        f"summary={args.output_dir/'summary.json'} table={args.output_dir/'summary_table_pretty.txt'}"
    )
    print(table, flush=True)


def build_contract(args: argparse.Namespace, samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
    candidate_manifest = args.candidate_cache.parent / "manifest.json"
    if not candidate_manifest.is_file():
        raise FileNotFoundError(candidate_manifest)
    candidate_contract = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    expected = {
        "type": "rag2_pced_exact_semantic_labeled_dynamic_topk_v2",
        "rows": 6545,
        "rerank_top_k": args.top_k,
        "evaluation_top_k": args.top_k,
        "candidate_layout": "source_balanced",
        "prompt_profile": "paper_compatible_three_anchor",
    }
    mismatch = {key: (value, candidate_contract.get(key)) for key, value in expected.items() if candidate_contract.get(key) != value}
    if mismatch:
        raise RuntimeError(f"Candidate contract mismatch: {mismatch}")
    return {
        "run_version": RUN_VERSION,
        "history_rule": HISTORY_RULE,
        "hypothesis": "past selected-token support reduces unstable expert switching without material accuracy loss",
        "questions": len(samples),
        "selected_ids_sha256": canonical_hash([sample.id for sample in samples]),
        "questions_per_dataset": args.questions_per_dataset,
        "fidelity_per_dataset": args.fidelity_per_dataset,
        "top_k": args.top_k,
        "candidate_cache": {"path": str(args.candidate_cache.resolve()), "sha256": sha256_file(args.candidate_cache)},
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "semantic_score_cache": {"path": str(args.semantic_score_cache.resolve()), "sha256": sha256_file(args.semantic_score_cache)},
        "legacy_output_dir": str(args.legacy_output_dir.resolve()),
        "llama": model_identity(args.llama_model),
        "prompt_and_parser": "identical anchored three-anchor Rationale+Answer contract",
        "candidate_and_document_order": "exact stored Top-8 reranked documents; unchanged",
        "gamma": args.gamma,
        "prior_epsilon": args.prior_epsilon,
        "history_strength": args.history_strength,
        "history_decay": args.history_decay,
        "history_temperature": args.history_temperature,
        "history_cap": args.history_cap,
        "pced_question_batch_size": args.pced_question_batch_size,
        "plain_question_batch_size": args.plain_question_batch_size,
        "choice_prompt_batch_size": args.choice_prompt_batch_size,
        "shard_size": args.shard_size,
        "max_input_tokens": args.max_input_tokens,
        "max_rationale_tokens": args.max_rationale_tokens,
        "answer_reserve_tokens": args.answer_reserve_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "selection_policy": (
            "seeded uniform sample of N rows per dataset; generation batches ordered by legacy rationale "
            "length for scheduling efficiency only; no metric-based input or score change"
        ),
        "test_tuning": "none; one preregistered history setting in a bounded audit cohort",
        "script_sha256": sha256_file(Path(__file__)),
    }


def ensure_contract(args: argparse.Namespace, contract: dict[str, Any]) -> str:
    stable = dict(contract)
    stable.pop("script_sha256", None)
    contract_hash = canonical_hash(stable)
    path = args.output_dir / "experiment_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.pop("script_sha256", None)
        if previous != stable:
            raise RuntimeError(f"Pilot contract mismatch; use a versioned output directory: {path}")
    else:
        atomic_json(path, contract)
    return contract_hash


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k != 8:
        raise ValueError("This bounded pilot is frozen to Top-8")
    positive = (
        args.questions_per_dataset,
        args.fidelity_per_dataset,
        args.pced_question_batch_size,
        args.plain_question_batch_size,
        args.choice_prompt_batch_size,
        args.shard_size,
        args.bootstrap_replicates,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Counts, batch sizes, shard size and bootstrap replicates must be positive")
    if not 0.0 <= args.history_decay < 1.0:
        raise ValueError("history-decay must be in [0, 1)")
    if args.history_temperature <= 0 or args.history_cap <= 0 or args.history_strength < 0:
        raise ValueError("Invalid history hyperparameters")
    if args.max_input_tokens <= args.max_rationale_tokens + args.answer_reserve_tokens:
        raise ValueError("Generation reserves leave no prompt budget")
    for path in (
        args.candidate_cache,
        args.semantic_score_cache,
        args.llama_model / "config.json",
        args.legacy_output_dir / "experiment_manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    if args.no_rag_root is None:
        from evaluate_rag2_pced_rationale_answer import DEFAULT_NO_RAG

        args.no_rag_root = DEFAULT_NO_RAG
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stages = (
        "preflight and balanced cohort selection",
        "load one Llama engine",
        "new-engine fixed PCED rerank control and legacy fidelity",
        "reuse or regenerate No-RAG",
        "reuse or regenerate Base-RAG",
        "reuse or regenerate semantic PCED",
        "history-conditioned rerank PCED",
        "paired metrics and routing diagnostics",
    )
    progress = PilotProgress(stages)
    generator: BatchedGenerator | None = None
    try:
        progress.start(1, 6, unit="check")
        # load_inputs validates exact benchmark/candidate row alignment.  Do
        # not let its legacy max-questions option truncate before balancing.
        args.max_questions = 0
        all_samples, all_candidates = load_inputs(args)
        progress.update(1)
        samples, candidates = select_balanced(
            all_samples, all_candidates, args.questions_per_dataset, args.seed
        )
        progress.update(1)
        semantic = load_semantic(args.semantic_score_cache)
        expected_pairs = {
            pair_key(sample, document)
            for sample, row in zip(samples, candidates, strict=True)
            for document in row["reranked_documents"]
        }
        if not expected_pairs.issubset(semantic):
            missing = expected_pairs - set(semantic)
            raise RuntimeError(f"Semantic score cache misses {len(missing)} pilot pairs; first={next(iter(missing))}")
        progress.update(1)
        cached_no_rag = load_no_rag(args, samples)
        progress.update(1)
        selected_ids = {sample.id for sample in samples}
        legacy = {
            name: legacy_condition(args.legacy_output_dir, name, selected_ids)
            for name in ("base_rag", "pced_rerank", "pced_semantic")
        }
        progress.update(1)
        # Group questions with similar observed legacy rationale lengths.  This
        # is scheduling metadata only: it never enters model inputs or scores.
        schedule = sorted(
            range(len(samples)),
            key=lambda index: (
                int(legacy["pced_rerank"][samples[index].id].get("rationale_tokens") or 0),
                samples[index].dataset,
                samples[index].row_idx,
            ),
        )
        samples = [samples[index] for index in schedule]
        candidates = [candidates[index] for index in schedule]
        contract = build_contract(args, samples)
        contract["legacy_selected_output_hashes"] = {
            name: canonical_hash([
                {
                    "sample_id": sample.id,
                    "answer": legacy[name][sample.id].get("answer"),
                    "rationale": legacy[name][sample.id].get("rationale"),
                    "rationale_tokens": legacy[name][sample.id].get("rationale_tokens"),
                }
                for sample in samples
            ])
            for name in ("base_rag", "pced_rerank", "pced_semantic")
        }
        contract["cached_no_rag_selected_output_hash"] = canonical_hash([
            {
                "sample_id": sample.id,
                "answer": cached_no_rag[sample.id].get("answer"),
                "rationale": cached_no_rag[sample.id].get("rationale"),
            }
            for sample in samples
        ])
        contract_hash = ensure_contract(args, contract)
        progress.update(1)
        progress.complete(
            f"questions={len(samples)} ({args.questions_per_dataset} x {len(DATASETS)} datasets) "
            f"documents={len(samples)*args.top_k} manifest={args.output_dir/'experiment_manifest.json'}"
        )
        if args.preflight_only:
            print("[preflight-only complete] no model was loaded and no generation was run", flush=True)
            return

        progress.start(2, 1, unit="model")
        generator = BatchedGenerator(args)
        progress.update(1)
        allocated = torch.cuda.max_memory_allocated(generator.device) / (1024**3)
        progress.complete(f"model={args.llama_model} initial_peak_vram={allocated:.2f}GiB")

        rerank_rows = run_cached_condition(
            args=args,
            condition="pced_rerank",
            samples=samples,
            candidates=candidates,
            semantic=semantic,
            generator=generator,
            contract_hash=contract_hash,
            progress=progress,
            stage_index=3,
            future_question_equivalents=len(samples),
        )
        fidelity_ids = selected_fidelity_ids(samples, args.fidelity_per_dataset)
        fidelity = fidelity_report(samples, rerank_rows, legacy["pced_rerank"], fidelity_ids)
        atomic_json(args.output_dir / "legacy_fidelity.json", fidelity)
        print(
            f"[fidelity] N={fidelity['questions']} answer={fidelity['answer_matches']}/{fidelity['questions']} "
            f"rationale_exact={fidelity['rationale_exact_matches']}/{fidelity['questions']} "
            f"token_count={fidelity['rationale_token_count_matches']}/{fidelity['questions']} "
            f"exact_pass={fidelity['exact_pass']}",
            flush=True,
        )

        if fidelity["exact_pass"]:
            no_rag_rows = [cached_no_rag[sample.id] for sample in samples]
            base_rows = [legacy["base_rag"][sample.id] for sample in samples]
            semantic_rows = [legacy["pced_semantic"][sample.id] for sample in samples]
            baseline_source = "legacy exact outputs reused after exact fixed-PCED fidelity"
            for stage_index, condition in ((4, "no_rag"), (5, "base_rag"), (6, "pced_semantic")):
                progress.start(stage_index, len(samples), initial=len(samples), detail=condition)
                progress.complete(f"exact fidelity passed; reused {condition} legacy output")
        else:
            print(
                "[fidelity decision] exact legacy match failed; regenerating all remaining baselines with "
                "the same batched engine. The new-engine rerank control remains the causal baseline.",
                flush=True,
            )
            # Every baseline is separately resumable and reports its own
            # active-stage rate and ETA.
            no_rag_rows = run_cached_condition(
                args=args, condition="no_rag", samples=samples, candidates=candidates,
                semantic=semantic, generator=generator, contract_hash=contract_hash,
                progress=progress, stage_index=4, future_question_equivalents=None,
            )
            base_rows = run_cached_condition(
                args=args, condition="base_rag", samples=samples, candidates=candidates,
                semantic=semantic, generator=generator, contract_hash=contract_hash,
                progress=progress, stage_index=5, future_question_equivalents=None,
            )
            semantic_rows = run_cached_condition(
                args=args, condition="pced_semantic", samples=samples, candidates=candidates,
                semantic=semantic, generator=generator, contract_hash=contract_hash,
                progress=progress, stage_index=6, future_question_equivalents=len(samples),
            )
            baseline_source = "all conditions regenerated with batched engine after legacy mismatch"

        history_rows = run_cached_condition(
            args=args,
            condition="pced_history",
            samples=samples,
            candidates=candidates,
            semantic=semantic,
            generator=generator,
            contract_hash=contract_hash,
            progress=progress,
            stage_index=7,
            future_question_equivalents=0,
        )
        conditions = {
            "no_rag": no_rag_rows,
            "base_rag": base_rows,
            "pced_rerank": rerank_rows,
            "pced_semantic": semantic_rows,
            "pced_history": history_rows,
        }
        aggregate(args, samples, conditions, fidelity, baseline_source, progress)
        peak = torch.cuda.max_memory_allocated(generator.device) / (1024**3)
        print(
            f"[workflow complete | elapsed {format_duration(time.time()-progress.started)} | "
            f"peak_vram={peak:.2f}GiB] output={args.output_dir}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[workflow FAILED | stage={progress.index}/{len(stages)} | completed={progress.done}/{progress.total} | "
            f"elapsed={format_duration(time.time()-progress.started)}] {type(exc).__name__}: {exc}; "
            "rerun the identical command to resume from complete atomic shards",
            flush=True,
        )
        raise
    finally:
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    main()

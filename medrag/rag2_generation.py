from __future__ import annotations

import math
import re
from typing import Any

from .rag2_mcq import parse_mcq_output


PPL_SCOPE_VERSION = "rag2_dual_generated_span_ppl_v1"
GENERATION_POLICY_VERSION = "rag2_compact_384_retry_v2"
FIXED_TARGET_PPL_VERSION = "rag2_fixed_no_doc_rationale_teacher_forced_v1"


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return re.sub(r"(?is)(<\|im_start\|>assistant\s*)<think>\s*$", r"\1", str(rendered))
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


def selected_logprob(logprob_row: Any, token_id: int) -> float | None:
    if logprob_row is None:
        return None
    item = None
    if isinstance(logprob_row, dict):
        item = logprob_row.get(token_id)
        if item is None:
            item = logprob_row.get(str(token_id))
        if item is None and len(logprob_row) == 1:
            item = next(iter(logprob_row.values()))
    if item is None:
        return None
    try:
        return float(getattr(item, "logprob", item))
    except (TypeError, ValueError):
        return None


def span_stats(logprobs: list[float]) -> dict[str, float | int | None]:
    valid = [float(value) for value in logprobs if value is not None and math.isfinite(float(value))]
    if not valid:
        return {"token_count": 0, "cumulative_logprob": None, "avg_neg_logprob": None, "ppl": None}
    cumulative = float(sum(valid))
    avg_neg_logprob = float(-cumulative / len(valid))
    return {
        "token_count": len(valid),
        "cumulative_logprob": cumulative,
        "avg_neg_logprob": avg_neg_logprob,
        "ppl": float(math.exp(min(50.0, avg_neg_logprob))),
    }


def teacher_forced_target_encoding(
    tokenizer: Any,
    target_text: str,
    options: dict[str, str],
) -> dict[str, Any]:
    """Tokenize one fixed paper-style response and retain its scoring spans."""
    parsed = parse_mcq_output(target_text, options)
    if parsed.parse_errors or parsed.rationale_only_span is None or parsed.answer_conclusion_span is None:
        raise ValueError(f"Fixed target is not a valid paper-style rationale: {parsed.parse_errors}")
    encoded = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]

    def overlapping(span: tuple[int, int]) -> list[int]:
        start, end = span
        return [idx for idx, (left, right) in enumerate(offsets) if right > start and left < end]

    return {
        "text": target_text,
        "token_ids": token_ids,
        "rationale_only_indices": overlapping(parsed.rationale_only_span),
        "answer_conclusion_indices": overlapping(parsed.answer_conclusion_span),
    }


def build_teacher_forced_request(
    tokenizer: Any,
    prompt: str,
    target: dict[str, Any],
    max_model_len: int,
) -> dict[str, Any]:
    prompt_ids = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
    target_ids = list(target["token_ids"])
    input_ids = prompt_ids + target_ids
    if len(input_ids) + 1 > max_model_len:
        raise ValueError(
            f"Teacher-forced sequence exceeds model limit: prompt={len(prompt_ids)} "
            f"target={len(target_ids)} max_model_len={max_model_len}"
        )
    return {
        "input_ids": input_ids,
        "prompt_token_count": len(prompt_ids),
        "target": target,
    }


def score_teacher_forced_output(output: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Score only the fixed target suffix of a vLLM prompt-logprob request."""
    input_ids = request["input_ids"]
    returned_ids = [int(token_id) for token_id in (getattr(output, "prompt_token_ids", None) or [])]
    if returned_ids != input_ids:
        raise RuntimeError("vLLM returned prompt token IDs that differ from the teacher-forced request.")
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None or len(prompt_logprobs) != len(input_ids):
        raise RuntimeError("vLLM did not return complete prompt logprobs for teacher forcing.")

    prompt_count = int(request["prompt_token_count"])
    target_ids = request["target"]["token_ids"]
    token_logprobs: list[float] = []
    for local_index, token_id in enumerate(target_ids):
        value = selected_logprob(prompt_logprobs[prompt_count + local_index], int(token_id))
        if value is None or not math.isfinite(value):
            raise RuntimeError(f"Missing target-token logprob at target index {local_index}")
        token_logprobs.append(float(value))

    rationale_indices = request["target"]["rationale_only_indices"]
    answer_indices = request["target"]["answer_conclusion_indices"]
    return {
        "rationale_with_answer": span_stats(token_logprobs),
        "rationale_only": span_stats([token_logprobs[idx] for idx in rationale_indices]),
        "answer_conclusion": span_stats([token_logprobs[idx] for idx in answer_indices]),
        "prompt_token_count": prompt_count,
        "target_token_count": len(target_ids),
    }


def finite_span_ppl(stats: dict[str, Any], scope: str) -> float | None:
    value = ((stats.get(scope) or {}).get("ppl"))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def teacher_forced_ppl_delta(
    no_doc_stats: dict[str, Any],
    with_doc_stats: dict[str, Any],
    scope: str,
) -> float | None:
    no_doc = finite_span_ppl(no_doc_stats, scope)
    with_doc = finite_span_ppl(with_doc_stats, scope)
    return None if no_doc is None or with_doc is None else no_doc - with_doc


def generated_span_token_stats(
    output: Any,
    tokenizer: Any,
    generated_text: str,
    options: dict[str, str],
    span_attribute: str,
    *,
    parsed_output: Any | None = None,
) -> tuple[dict[str, Any], str]:
    token_ids = list(getattr(output, "token_ids", None) or [])
    token_logprob_rows = list(getattr(output, "logprobs", None) or [])
    if not token_ids or not token_logprob_rows:
        return span_stats([]), "unavailable_no_token_logprobs"

    # Inline supporting-evidence traces append a machine-readable section after
    # the terminal answer.  Their rationale spans are parsed from the response
    # prefix, while offsets still refer to the complete generated text.
    parsed = parsed_output if parsed_output is not None else parse_mcq_output(generated_text, options)
    target_span = getattr(parsed, span_attribute, None)
    if target_span is None:
        return span_stats([]), f"unavailable_no_{span_attribute}"

    start, end = target_span
    try:
        encoded = tokenizer(parsed.visible_text, add_special_tokens=False, return_offsets_mapping=True)
        encoded_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
    except Exception:
        encoded_ids, offsets = [], []

    best_shift = 0
    best_matches = -1
    # A generation prompt may prefill a short response marker such as "Rationale:".
    # Search a slightly wider offset so generated-token logprobs still align to the rationale span.
    for shift in range(-16, 17):
        matches = 0
        for encoded_idx, token_id in enumerate(encoded_ids):
            output_idx = encoded_idx + shift
            if 0 <= output_idx < len(token_ids) and int(token_ids[output_idx]) == int(token_id):
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_shift = shift

    rationale_positions = [
        index
        for index, (offset_start, offset_end) in enumerate(offsets)
        if int(offset_end) > start and int(offset_start) < end
    ]
    selected: list[float] = []
    matched_rationale_tokens = 0
    for encoded_idx in rationale_positions:
        output_idx = encoded_idx + best_shift
        if not (0 <= output_idx < len(token_ids)):
            continue
        if int(token_ids[output_idx]) != int(encoded_ids[encoded_idx]):
            continue
        logprob = selected_logprob(token_logprob_rows[output_idx], int(token_ids[output_idx]))
        if logprob is not None:
            selected.append(float(logprob))
            matched_rationale_tokens += 1

    if rationale_positions and matched_rationale_tokens / len(rationale_positions) >= 0.98:
        return span_stats(selected), "vllm_generated_token_logprobs_offset_aligned"

    reconstructed = tokenizer.decode(token_ids, skip_special_tokens=True)
    reconstructed_span = (
        target_span
        if parsed_output is not None
        else getattr(parse_mcq_output(reconstructed, options), span_attribute, None)
    )
    if reconstructed_span is None:
        return span_stats([]), "unavailable_token_alignment"
    fallback_start, fallback_end = reconstructed_span
    previous = ""
    fallback_values: list[float] = []
    for index, (token_id, logprob_row) in enumerate(zip(token_ids, token_logprob_rows), start=1):
        current = tokenizer.decode(token_ids[:index], skip_special_tokens=True)
        if not current.startswith(previous):
            return span_stats([]), "unavailable_non_monotonic_cumulative_decode"
        token_start = len(previous)
        token_end = len(current)
        logprob = selected_logprob(logprob_row, int(token_id))
        if token_end > fallback_start and token_start < fallback_end and logprob is not None:
            fallback_values.append(float(logprob))
        previous = current
    return span_stats(fallback_values), "vllm_generated_token_logprobs_cumulative_decode_fallback"


def rationale_token_stats(
    output: Any,
    tokenizer: Any,
    generated_text: str,
    options: dict[str, str],
    *,
    parsed_output: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """PPL over the paper-style rationale query, including its answer conclusion."""
    return generated_span_token_stats(
        output,
        tokenizer,
        generated_text,
        options,
        "rationale_span",
        parsed_output=parsed_output,
    )


def generation_stats(
    output: Any,
    tokenizer: Any,
    generated_text: str,
    options: dict[str, str],
    *,
    parsed_output: Any | None = None,
) -> dict[str, Any]:
    parsed_text = parsed_output if parsed_output is not None else parse_mcq_output(generated_text, options)
    token_ids = list(getattr(output, "token_ids", None) or [])
    full_logprob = getattr(output, "cumulative_logprob", None)
    if full_logprob is not None and token_ids:
        full_avg_neg_logprob = float(-float(full_logprob) / len(token_ids))
        full = {
            "token_count": len(token_ids),
            "cumulative_logprob": float(full_logprob),
            "avg_neg_logprob": full_avg_neg_logprob,
            "ppl": float(math.exp(min(50.0, full_avg_neg_logprob))),
        }
    else:
        full = {"token_count": len(token_ids), "cumulative_logprob": None, "avg_neg_logprob": None, "ppl": None}

    rationale, source = rationale_token_stats(
        output,
        tokenizer,
        generated_text,
        options,
        parsed_output=parsed_text,
    )
    rationale_only, rationale_only_source = generated_span_token_stats(
        output,
        tokenizer,
        generated_text,
        options,
        "rationale_only_span",
        parsed_output=parsed_text,
    )
    answer_conclusion, answer_conclusion_source = generated_span_token_stats(
        output,
        tokenizer,
        generated_text,
        options,
        "answer_conclusion_span",
        parsed_output=parsed_text,
    )
    rationale_with_answer = {
        **rationale,
        "char_start": None if parsed_text.rationale_span is None else parsed_text.rationale_span[0],
        "char_end": None if parsed_text.rationale_span is None else parsed_text.rationale_span[1],
        "source": source,
    }
    return {
        "full_generation": full,
        "rationale": rationale_with_answer,
        "rationale_plus_answer": rationale_with_answer,
        "rationale_only": {
            **rationale_only,
            "char_start": None if parsed_text.rationale_only_span is None else parsed_text.rationale_only_span[0],
            "char_end": None if parsed_text.rationale_only_span is None else parsed_text.rationale_only_span[1],
            "source": rationale_only_source,
        },
        "answer_conclusion": {
            **answer_conclusion,
            "char_start": (
                None if parsed_text.answer_conclusion_span is None else parsed_text.answer_conclusion_span[0]
            ),
            "char_end": (
                None if parsed_text.answer_conclusion_span is None else parsed_text.answer_conclusion_span[1]
            ),
            "source": answer_conclusion_source,
        },
    }


def flatten_generation_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = stats or {}
    full = stats.get("full_generation") or {}
    rationale = stats.get("rationale") or {}
    rationale_only = stats.get("rationale_only") or {}
    answer_conclusion = stats.get("answer_conclusion") or {}
    return {
        "token_count": full.get("token_count", 0),
        "cumulative_logprob": full.get("cumulative_logprob"),
        "avg_neg_logprob": full.get("avg_neg_logprob"),
        "ppl": full.get("ppl"),
        "ppl_scope": "rationale_plus_answer_conclusion_with_full_generation_fallback",
        "rationale_token_count": rationale.get("token_count", 0),
        "rationale_cumulative_logprob": rationale.get("cumulative_logprob"),
        "rationale_avg_neg_logprob": rationale.get("avg_neg_logprob"),
        "rationale_ppl": rationale.get("ppl"),
        "rationale_char_start": rationale.get("char_start"),
        "rationale_char_end": rationale.get("char_end"),
        "rationale_ppl_source": rationale.get("source"),
        "rationale_with_answer_token_count": rationale.get("token_count", 0),
        "rationale_with_answer_cumulative_logprob": rationale.get("cumulative_logprob"),
        "rationale_with_answer_avg_neg_logprob": rationale.get("avg_neg_logprob"),
        "rationale_with_answer_ppl": rationale.get("ppl"),
        "rationale_only_token_count": rationale_only.get("token_count", 0),
        "rationale_only_cumulative_logprob": rationale_only.get("cumulative_logprob"),
        "rationale_only_avg_neg_logprob": rationale_only.get("avg_neg_logprob"),
        "rationale_only_ppl": rationale_only.get("ppl"),
        "rationale_only_ppl_source": rationale_only.get("source"),
        "answer_conclusion_token_count": answer_conclusion.get("token_count", 0),
        "answer_conclusion_cumulative_logprob": answer_conclusion.get("cumulative_logprob"),
        "answer_conclusion_avg_neg_logprob": answer_conclusion.get("avg_neg_logprob"),
        "answer_conclusion_ppl": answer_conclusion.get("ppl"),
        "answer_conclusion_ppl_source": answer_conclusion.get("source"),
        "nested": stats,
    }


def preferred_rationale_ppl(stats: dict[str, Any] | None) -> float | None:
    flattened = flatten_generation_stats(stats)
    for key in ("rationale_ppl", "ppl"):
        value = flattened.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None

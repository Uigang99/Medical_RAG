from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .rag2_mcq import PAPER_EXACT_RATIONALE_INSTRUCTION, clean_text, format_question, normalized_options


TRACE_VERSION = "rag2_paper_compatible_three_anchor_v1"
PROMPT_VERSION = "rag2_paper_compatible_three_anchor_prompt_v1"
GENERATION_POLICY_VERSION = "rag2_three_anchor_rationale_then_constrained_choice_v1"

RATIONALE_LABEL = "Rationale:"
RATIONALE_HEADER = f"{RATIONALE_LABEL}\n"
END_REASONING_MARKER = "### END OF REASONING ###"
FINAL_ANSWER_PREFIX = "Final answer: ("
CHOICES = ("A", "B", "C", "D")
ANCHOR_NAMES = ("pre_rationale", "post_rationale", "pre_choice")

FORMAT_INSTRUCTION = (
    "Use exactly this response structure:\n"
    "Rationale:\n"
    "<step-by-step reasoning>\n"
    f"{END_REASONING_MARKER}\n"
    "Final answer: (<OPTION LETTER>) <EXACT OPTION TEXT>\n"
    "Do not state the final option letter before the Final answer line. "
    "Do not write anything after the final answer."
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_anchored_user_prompt(row: dict[str, Any], document_text: str | None = None) -> str:
    prompt = (
        f"{PAPER_EXACT_RATIONALE_INSTRUCTION}\n"
        f"{FORMAT_INSTRUCTION}\n"
        f"Here is the question: {format_question(row)}"
    )
    evidence = str(document_text or "").strip()
    if evidence:
        prompt += f"\n\nDocuments:\n{evidence}"
    return prompt


def render_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return re.sub(r"(?is)(<\|im_start\|>assistant\s*)<think>\s*$", r"\1", str(rendered))


def rationale_generation_prompt(tokenizer: Any, row: dict[str, Any], document_text: str | None) -> str:
    user_prompt = build_anchored_user_prompt(row, document_text)
    return render_chat_prompt(tokenizer, user_prompt) + RATIONALE_HEADER


def normalize_rationale(raw_text: str) -> tuple[str, list[str]]:
    """Return semantic rationale text without silently rewriting its content."""
    text = str(raw_text or "").strip()
    flags: list[str] = []
    if text.startswith(RATIONALE_LABEL):
        flags.append("repeated_rationale_header")
    if END_REASONING_MARKER in text:
        flags.append("embedded_end_reasoning_marker")
        text = text.split(END_REASONING_MARKER, 1)[0].rstrip()
    terminal_match = re.search(r"(?im)^\s*(?:Final answer|Therefore,\s*the answer is)\s*[: ]", text)
    if terminal_match:
        flags.append("embedded_final_answer")
        text = text[: terminal_match.start()].rstrip()
    if not text:
        flags.append("empty_rationale")
    return text, flags


def assistant_decision_prefix(rationale: str) -> str:
    return (
        f"{RATIONALE_HEADER}{str(rationale).strip()}\n"
        f"{END_REASONING_MARKER}\n"
        f"{FINAL_ANSWER_PREFIX}"
    )


def canonical_response(rationale: str, answer: str, options: dict[str, str]) -> str:
    label = str(answer).strip().upper()
    if label not in CHOICES or label not in options:
        raise ValueError(f"Invalid constrained answer: {answer!r}")
    return assistant_decision_prefix(rationale) + f"{label}) {clean_text(options[label])}"


def semantic_retrieval_queries(rationale: str, answer: str, options: dict[str, str]) -> dict[str, str]:
    label = str(answer).strip().upper()
    if label not in CHOICES or label not in options:
        raise ValueError(f"Invalid constrained answer: {answer!r}")
    rationale_only = str(rationale).strip()
    answer_line = f"Final answer: ({label}) {clean_text(options[label])}"
    return {
        "question_only": "",
        "rationale_only": rationale_only,
        "rationale_answer": f"{rationale_only}\n\n{answer_line}".strip(),
    }


def _last_overlapping_token(offsets: Sequence[Sequence[int]], span: tuple[int, int], name: str) -> int:
    start, end = span
    matches = [
        index
        for index, pair in enumerate(offsets)
        if int(pair[1]) > start and int(pair[0]) < end
    ]
    if not matches:
        raise RuntimeError(f"No tokenizer token overlaps {name} character span {span}")
    return matches[-1]


@dataclass(frozen=True)
class AnchoredEncoding:
    input_ids: list[int]
    anchor_indices: dict[str, int]
    anchor_token_ids: dict[str, int]
    anchor_token_text: dict[str, str]
    full_prefix_text: str
    prompt_sha256: str


def encode_to_pre_choice(
    tokenizer: Any,
    row: dict[str, Any],
    document_text: str | None,
    rationale: str,
) -> AnchoredEncoding:
    user_prompt = build_anchored_user_prompt(row, document_text)
    chat_prompt = render_chat_prompt(tokenizer, user_prompt)
    assistant = assistant_decision_prefix(rationale)
    full_text = chat_prompt + assistant
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]

    assistant_start = len(chat_prompt)
    rationale_label_start = assistant_start
    rationale_label_end = rationale_label_start + len(RATIONALE_LABEL)
    end_marker_start = full_text.rfind(END_REASONING_MARKER)
    final_prefix_start = full_text.rfind(FINAL_ANSWER_PREFIX)
    if end_marker_start < assistant_start or final_prefix_start < end_marker_start:
        raise RuntimeError("Anchored response markers are not in canonical order")
    spans = {
        "pre_rationale": (rationale_label_start, rationale_label_end),
        "post_rationale": (end_marker_start, end_marker_start + len(END_REASONING_MARKER)),
        "pre_choice": (
            final_prefix_start + len(FINAL_ANSWER_PREFIX) - 1,
            final_prefix_start + len(FINAL_ANSWER_PREFIX),
        ),
    }
    indices = {
        name: _last_overlapping_token(offsets, spans[name], name)
        for name in ANCHOR_NAMES
    }
    if not (indices["pre_rationale"] < indices["post_rationale"] < indices["pre_choice"]):
        raise RuntimeError(f"Anchor token order is invalid: {indices}")
    token_ids = {name: input_ids[index] for name, index in indices.items()}
    token_text = {
        name: tokenizer.decode([token_id], skip_special_tokens=False)
        for name, token_id in token_ids.items()
    }
    return AnchoredEncoding(
        input_ids=input_ids,
        anchor_indices=indices,
        anchor_token_ids=token_ids,
        anchor_token_text=token_text,
        full_prefix_text=full_text,
        prompt_sha256=sha256_text(user_prompt),
    )


def normalized_mcq_row(row: dict[str, Any]) -> dict[str, Any]:
    options = normalized_options(row)
    answer = row.get("answer", row.get("gold_answer"))
    if isinstance(answer, int) and 0 <= answer < len(CHOICES):
        answer = CHOICES[answer]
    answer = str(answer or "").strip().upper()
    if answer not in options:
        answers = row.get("answers")
        if isinstance(answers, list):
            candidates = [str(value).strip().upper() for value in answers]
            answer = next((value for value in candidates if value in options), "")
    if answer not in CHOICES or len(options) != 4:
        raise ValueError(f"Invalid four-choice row: answer={answer!r} options={sorted(options)}")
    return {
        **row,
        "question": clean_text(row.get("question")),
        "options": {choice: options[choice] for choice in CHOICES},
        "answer": answer,
    }

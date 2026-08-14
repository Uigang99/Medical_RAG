from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any


PROMPT_VERSION = "rag2_mcq_rationale_paper_focused_v4"
DOCUMENT_PROMPT_VERSION = "rag2_mcq_document_rationale_paper_focused_v3"
PAPER_EXACT_PROMPT_VERSION = "rag2_llama3_rationale_paper_exact_v1"
PAPER_EXACT_DOCUMENT_PROMPT_VERSION = "rag2_llama3_rationale_paper_exact_document_v1"
PAPER_EXACT_TERMINAL_PROMPT_VERSION = "rag2_llama3_rationale_paper_exact_terminal_v1"
PAPER_EXACT_TERMINAL_DOCUMENT_PROMPT_VERSION = "rag2_llama3_rationale_paper_exact_terminal_document_v1"
PAPER_ANSWER_FORMAT_PROMPT_VERSION = "rag2_llama3_rationale_paper_answer_format_v2"
PAPER_ANSWER_FORMAT_DOCUMENT_PROMPT_VERSION = "rag2_llama3_rationale_paper_answer_format_document_v3"

PAPER_EXACT_RATIONALE_INSTRUCTION = (
    "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, "
    "starting by summarizing the available information. Output your explanation and single option from the given "
    "options as the final answer."
)

# This is deliberately the only sentence added to the prompt reported in
# RAG2.  It constrains serialization, not the medical reasoning, length, use
# of evidence, or answer-selection policy.
PAPER_EXACT_TERMINAL_FORMAT_INSTRUCTION = (
    "End the response with exactly one final line in this format: "
    "Therefore, the answer is (<OPTION LETTER>) <EXACT OPTION TEXT>."
)

PAPER_RATIONALE_INSTRUCTION = (
    "The following are multiple choice questions about medical knowledge. Select the single best option and provide "
    "a concise medical rationale for your answer."
)

FIGURE4_COMPACT_RATIONALE_GUIDANCE = (
    "Write one focused, coherent paragraph centered on the decisive medical evidence supporting the selected answer. "
    "You may briefly contrast another option only when that distinction is necessary to establish the best answer. "
    "Do not review the options one by one, repeat the question, add tangential background teaching, reconsider your "
    "decision, or criticize the question. Do not use headings or bullet points."
)

RETRIEVED_EVIDENCE_INSTRUCTION = (
    "A retrieved document is provided below as additional context for the question. Keep the rationale focused on the "
    "medical reasoning that supports the selected answer rather than on discussing, summarizing, evaluating, or citing "
    "the document itself."
)

DECISIVE_REPAIR_RATIONALE_GUIDANCE = (
    "Write one focused, coherent paragraph containing only the decisive medical finding and the direct reason it "
    "supports the selected option. Omit all alternative-option analysis, caveats, question criticism, background "
    "teaching, headings, and bullet points."
)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def render_paper_document_view(document: dict[str, Any], max_doc_chars: int = 2600) -> str:
    """Render the exact document view appended to paper-style prompts."""
    title = clean_text(document.get("title"))
    body = clean_text(document.get("text"))
    document_text = "\n".join(part for part in (title, body) if part)
    if max_doc_chars > 0 and len(document_text) > max_doc_chars:
        document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
    return document_text


def normalized_options(row: dict[str, Any]) -> dict[str, str]:
    options = row.get("options")
    if isinstance(options, dict):
        return {str(key).upper(): clean_text(value) for key, value in options.items()}

    choices = row.get("choices")
    if isinstance(choices, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {labels[idx]: clean_text(value) for idx, value in enumerate(choices) if idx < len(labels)}
    return {}


def format_question(row: dict[str, Any]) -> str:
    options = normalized_options(row)
    lines = [clean_text(row.get("question"))]
    lines.extend(f"{label}. {options[label]}" for label in sorted(options))
    return "\n".join(line for line in lines if line)


def _generation_instruction(
    row: dict[str, Any],
    format_retry: bool,
    selected_answer: str | None = None,
    compact_retry: bool = False,
    use_retrieved_evidence: bool = False,
) -> str:
    selected_answer = str(selected_answer or "").upper()
    selected_option_text = clean_text(normalized_options(row).get(selected_answer))
    answer_anchor = (
        f"A prior answer-selection pass fixed the answer as ({selected_answer}) {selected_option_text}. "
        "Do not reconsider or change that option. State only the decisive medical fact that supports it, without "
        "discussing alternatives, ambiguity, or flaws in the question. "
        if selected_answer in normalized_options(row)
        else ""
    )
    retry_instruction = (
        "The previous response could not be parsed. Follow the requested final-answer format exactly and select one of "
        "the given options. "
        if format_retry
        else ""
    )
    compact_retry_instruction = (
        "The previous response exceeded the available output space. Answer again more compactly, retaining only the "
        "decisive medical evidence and conclusion. Do not compare alternatives, critique the question, revisit the "
        "decision, or add background teaching. "
        if compact_retry
        else ""
    )
    rationale_guidance = (
        DECISIVE_REPAIR_RATIONALE_GUIDANCE
        if compact_retry or selected_answer in normalized_options(row)
        else FIGURE4_COMPACT_RATIONALE_GUIDANCE
    )
    evidence_instruction = f" {RETRIEVED_EVIDENCE_INSTRUCTION}" if use_retrieved_evidence else ""
    return (
        retry_instruction
        + compact_retry_instruction
        + answer_anchor
        + PAPER_RATIONALE_INSTRUCTION
        + evidence_instruction
        + " "
        + rationale_guidance
        + " End your explanation with: 'Therefore, the answer is (<option letter>) <option text>.'"
    )


def build_no_rag_messages(
    row: dict[str, Any],
    *,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    instruction = _generation_instruction(row, format_retry, selected_answer, compact_retry)
    return [{"role": "user", "content": f"{instruction}\n\nHere is the question:\n{format_question(row)}"}]


def build_paper_exact_no_rag_messages(
    row: dict[str, Any],
    *,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Build the rationale prompt reported verbatim in RAG2 section 3.3.

    Additional wording is used only by explicitly requested repair passes. The
    standard generation path remains the paper prompt without style or length
    constraints.
    """
    additions: list[str] = []
    options = normalized_options(row)
    selected_answer = str(selected_answer or "").upper()
    if format_retry:
        additions.append("Make the final answer clearly identify exactly one option from the given options.")
    if compact_retry:
        additions.append("Complete the reasoning and final answer within the available output space.")
    if selected_answer in options:
        additions.append(
            f"Use ({selected_answer}) {options[selected_answer]} as the selected final option and explain the reasoning."
        )

    instruction = PAPER_EXACT_RATIONALE_INSTRUCTION
    if additions:
        instruction = f"{instruction} {' '.join(additions)}"
    return [{"role": "user", "content": f"{instruction}\nHere is the question: {format_question(row)}"}]


def build_paper_exact_terminal_no_rag_messages(
    row: dict[str, Any],
    *,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Use the published RAG2 prompt plus one output-serialization rule.

    The compatibility arguments intentionally have no effect.  Repair is
    performed by constrained decoding rather than by changing the reasoning
    instruction between attempts.
    """
    del format_retry, selected_answer, compact_retry
    instruction = f"{PAPER_EXACT_RATIONALE_INSTRUCTION} {PAPER_EXACT_TERMINAL_FORMAT_INSTRUCTION}"
    return [{"role": "user", "content": f"{instruction}\nHere is the question: {format_question(row)}"}]


def build_paper_answer_format_no_rag_messages(
    row: dict[str, Any],
    *,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Use the reported RAG2 prompt with only a parseable final-sentence contract."""
    additions = [
        "After the explanation, write the final answer on a separate last line. "
        "That line must contain only this format: "
        "'Therefore, the answer is (<OPTION LETTER>) <EXACT OPTION TEXT>.' "
        "Replace the placeholders with one uppercase option letter and the corresponding option text copied verbatim. "
        "The parentheses around the option letter are mandatory. Do not write anything after this final line."
    ]
    options = normalized_options(row)
    selected_answer = str(selected_answer or "").upper()
    if format_retry:
        additions.append("The previous response did not follow the required final-sentence format.")
    if compact_retry:
        additions.append("Complete the reasoning and final answer within the available output space.")
    if selected_answer in options:
        additions.append(
            f"Use ({selected_answer}) {options[selected_answer]} as the selected final option and explain the reasoning."
        )

    instruction = f"{PAPER_EXACT_RATIONALE_INSTRUCTION} {' '.join(additions)}"
    return [{"role": "user", "content": f"{instruction}\nHere is the question: {format_question(row)}"}]


def build_paper_exact_documents_messages(
    row: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    max_doc_chars: int = 1800,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Append raw evidence chunks without changing the paper's rationale instruction.

    The RAG2 paper specifies the rationale prompt but not an extra behavioral
    instruction for evidence-conditioned answer generation.  This builder
    therefore keeps that prompt verbatim and merely appends the retrieved
    chunk text. Source labels, IDs, titles, and rank markers are deliberately
    omitted: they are metadata rather than evidence and can create corpus or
    position bias. Chunks remain separated by one blank line.
    Repair arguments are retained for a shared call interface; normal
    ``paper_exact`` runs deliberately leave all of them at their defaults.
    """
    base = build_paper_exact_no_rag_messages(
        row,
        format_retry=format_retry,
        selected_answer=selected_answer,
        compact_retry=compact_retry,
    )[0]["content"]
    rendered_documents: list[str] = []
    for document in documents:
        document_text = clean_text(document.get("text"))
        if max_doc_chars > 0 and len(document_text) > max_doc_chars:
            document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
        if document_text:
            rendered_documents.append(document_text)
    rendered_context = "\n\n".join(rendered_documents)
    return [{"role": "user", "content": f"{base}\n\nDocuments:\n{rendered_context}"}]


def build_paper_exact_terminal_documents_messages(
    row: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    max_doc_chars: int = 1800,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Append only raw evidence chunks to the fixed-terminal paper prompt."""
    base = build_paper_exact_terminal_no_rag_messages(
        row,
        format_retry=format_retry,
        selected_answer=selected_answer,
        compact_retry=compact_retry,
    )[0]["content"]
    rendered_documents: list[str] = []
    for document in documents:
        document_text = clean_text(document.get("text"))
        if max_doc_chars > 0 and len(document_text) > max_doc_chars:
            document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
        if document_text:
            rendered_documents.append(document_text)
    rendered_context = "\n\n".join(rendered_documents)
    return [{"role": "user", "content": f"{base}\n\nDocuments:\n{rendered_context}"}]


def build_paper_answer_format_document_messages(
    row: dict[str, Any],
    document: dict[str, Any],
    *,
    max_doc_chars: int = 2600,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Add one document to the paper answer-format prompt without extra guidance.

    The base task instruction and terminal-answer contract are identical to
    no-RAG. The document is merely part of the model input: the prompt neither
    encourages the model to ignore it nor forces it to use or cite it.
    """
    base = build_paper_answer_format_no_rag_messages(
        row,
        format_retry=format_retry,
        selected_answer=selected_answer,
        compact_retry=compact_retry,
    )[0]["content"]
    question_marker = "\nHere is the question: "
    if question_marker not in base:
        raise RuntimeError("Unexpected paper-answer prompt layout.")
    instruction, question = base.split(question_marker, 1)

    document_text = render_paper_document_view(document, max_doc_chars)

    content = (
        f"{instruction}\n"
        f"Here is the question: {question}\n\n"
        f"Document:\n{document_text}"
    )
    return [{"role": "user", "content": content}]


def build_paper_answer_format_documents_messages(
    row: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    max_doc_chars: int = 1800,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Append a ranked document list while preserving the paper no-RAG prompt.

    This is the final-answer counterpart to the single-document pseudo-label
    prompt: it adds evidence only, without instructing the model to blindly use
    it, ignore it, or cite it.  That keeps Top-k and oracle-context evaluation
    conditions comparable.
    """
    base = build_paper_answer_format_no_rag_messages(
        row,
        format_retry=format_retry,
        selected_answer=selected_answer,
        compact_retry=compact_retry,
    )[0]["content"]
    question_marker = "\nHere is the question: "
    if question_marker not in base:
        raise RuntimeError("Unexpected paper-answer prompt layout.")
    instruction, question = base.split(question_marker, 1)

    rendered_documents: list[str] = []
    for rank, document in enumerate(documents, start=1):
        title = clean_text(document.get("title"))
        body = clean_text(document.get("text"))
        document_text = "\n".join(part for part in (title, body) if part)
        if max_doc_chars > 0 and len(document_text) > max_doc_chars:
            document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
        stable_id = clean_text(
            document.get("stable_id")
            or document.get("corpus_id")
            or document.get("chunk_id")
            or document.get("db_id")
        )
        source = clean_text(document.get("source"))
        header = f"[{rank}] source={source} id={stable_id}".strip()
        rendered_documents.append("\n".join(part for part in (header, document_text) if part))

    rendered_context = "\n\n".join(rendered_documents)
    content = (
        f"{instruction}\n"
        f"Here is the question: {question}\n\n"
        f"Documents:\n{rendered_context}"
    )
    return [{"role": "user", "content": content}]


def build_choice_selection_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    valid_options = ", ".join(sorted(normalized_options(row)))
    return [
        {"role": "system", "content": "You select one answer choice for medical multiple-choice questions."},
        {
            "role": "user",
            "content": (
                "Choose the single best available option, even if the item is imperfect. "
                f"Return only one capital option letter from: {valid_options}.\n\nQuestion:\n{format_question(row)}"
            ),
        },
    ]


def build_document_messages(
    row: dict[str, Any],
    document: dict[str, Any],
    *,
    max_doc_chars: int = 2600,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    instruction = _generation_instruction(
        row,
        format_retry,
        selected_answer,
        compact_retry,
        use_retrieved_evidence=True,
    )
    title = clean_text(document.get("title"))
    body = clean_text(document.get("text"))
    document_text = "\n".join(part for part in (title, body) if part)
    if max_doc_chars > 0 and len(document_text) > max_doc_chars:
        document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
    user_content = f"{instruction}\n\nRetrieved document:\n{document_text}\n\nHere is the question:\n{format_question(row)}"
    return [{"role": "user", "content": user_content}]


def build_documents_messages(
    row: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    max_doc_chars: int = 2200,
    format_retry: bool = False,
    selected_answer: str | None = None,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Build the same RAG2 answer contract with a ranked multi-document context."""
    instruction = _generation_instruction(
        row,
        format_retry,
        selected_answer,
        compact_retry,
        use_retrieved_evidence=True,
    )
    rendered_documents: list[str] = []
    for rank, document in enumerate(documents, start=1):
        title = clean_text(document.get("title"))
        body = clean_text(document.get("text"))
        document_text = "\n".join(part for part in (title, body) if part)
        if max_doc_chars > 0 and len(document_text) > max_doc_chars:
            document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
        source = clean_text(document.get("source"))
        stable_id = clean_text(
            document.get("corpus_id")
            or document.get("chunk_id")
            or document.get("db_id")
            or document.get("stable_id")
        )
        rendered_documents.append(
            f"[{rank}] source={source} id={stable_id}\n{document_text}".strip()
        )

    context = "\n\n".join(rendered_documents)
    user_content = (
        f"{instruction}\n\nRetrieved documents:\n{context}\n\n"
        f"Here is the question:\n{format_question(row)}"
    )
    return [{"role": "user", "content": user_content}]


def build_document_choice_selection_messages(
    row: dict[str, Any],
    document: dict[str, Any],
    *,
    max_doc_chars: int = 2600,
) -> list[dict[str, str]]:
    """Select an answer for a format-repair fallback without exposing the gold label."""
    valid_options = ", ".join(sorted(normalized_options(row)))
    title = clean_text(document.get("title"))
    body = clean_text(document.get("text"))
    document_text = "\n".join(part for part in (title, body) if part)
    if max_doc_chars > 0 and len(document_text) > max_doc_chars:
        document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
    return [
        {"role": "system", "content": "You select one answer choice for medical multiple-choice questions."},
        {
            "role": "user",
            "content": (
                "Use the retrieved document and the question to choose the single best available option, even if the item "
                "is imperfect. "
                f"Return only one capital option letter from: {valid_options}.\n\n"
                f"Retrieved document:\n{document_text}\n\nQuestion:\n{format_question(row)}"
            ),
        },
    ]


def build_documents_choice_selection_messages(
    row: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    max_doc_chars: int = 2200,
) -> list[dict[str, str]]:
    """Select one option during format repair while retaining the multi-document context."""
    valid_options = ", ".join(sorted(normalized_options(row)))
    rendered_documents: list[str] = []
    for rank, document in enumerate(documents, start=1):
        title = clean_text(document.get("title"))
        body = clean_text(document.get("text"))
        document_text = "\n".join(part for part in (title, body) if part)
        if max_doc_chars > 0 and len(document_text) > max_doc_chars:
            document_text = document_text[: max_doc_chars - 3].rstrip() + "..."
        rendered_documents.append(f"[{rank}]\n{document_text}".strip())
    context = "\n\n".join(rendered_documents)
    return [
        {"role": "system", "content": "You select one answer choice for medical multiple-choice questions."},
        {
            "role": "user",
            "content": (
                "Use relevant retrieved evidence and medical knowledge to choose the single best available option. "
                f"Return only one capital option letter from: {valid_options}.\n\n"
                f"Retrieved documents:\n{context}\n\nQuestion:\n{format_question(row)}"
            ),
        },
    ]


def strip_hidden_thinking(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    return text.strip()


def paper_exact_terminal_line(options: dict[str, Any], answer: str) -> str:
    """Return the one canonical final line used by the controlled evaluator."""
    label = str(answer or "").upper()
    option_text = clean_text((options or {}).get(label))
    if not option_text:
        raise ValueError(f"Cannot build terminal answer for option {label!r}.")
    suffix = "" if option_text.endswith((".", "?", "!")) else "."
    return f"Therefore, the answer is ({label}) {option_text}{suffix}"


def paper_exact_terminal_regex(options: dict[str, Any]) -> str:
    """Build a per-question grammar ending in one exact available option.

    vLLM/XGrammar treats the supplied regular expression as the complete
    generated response.  The prefix remains unconstrained free rationale,
    while the last line is restricted to a canonical option line.
    """

    def escape_literal(value: str) -> str:
        # Keep ordinary spaces and Unicode text readable.  Escape only regex
        # metacharacters supported by the grammar backend.
        return re.sub(r"([\\.^$|?*+(){}\[\]])", r"\\\1", value)

    lines = [
        escape_literal(paper_exact_terminal_line(options, label))
        for label in sorted(str(key).upper() for key in (options or {}))
    ]
    if not lines:
        raise ValueError("Cannot build a terminal-answer regex without options.")
    return rf"(.|\n)*\n({'|'.join(lines)})"


def append_paper_exact_terminal_answer(text: str, options: dict[str, Any], answer: str) -> str:
    """Append a canonical terminal line while preserving generated reasoning."""
    visible = strip_hidden_thinking(text)
    lines = visible.splitlines()
    # Drop only a trailing malformed attempt at the required serialization;
    # all medical reasoning produced by the model remains untouched.
    while lines and re.match(r"(?i)^\s*therefore\s*,?\s+the\s+answer\s+is\b", lines[-1]):
        lines.pop()
    reasoning = "\n".join(lines).rstrip()
    terminal = paper_exact_terminal_line(options, answer)
    return f"{reasoning}\n{terminal}" if reasoning else terminal


def canonicalize_rationale_query(
    rationale: str,
    final_answer: str | None,
    options: dict[str, Any] | None,
) -> str:
    """Keep the retrieval query in the paper-style answer-conclusion form."""
    cleaned = clean_text(rationale)
    option_text = clean_text((options or {}).get(str(final_answer or "").upper()))
    if not cleaned or not final_answer or not option_text:
        return cleaned

    conclusion = re.search(
        r"(?is)\btherefore\s*,?\s+the\s+answer\s+is\s+(?:\(\s*)?[A-Za-z](?:\s*\))?"
        r"(?:\s*[.:]\s*|\s+).*?\s*$",
        cleaned,
    )
    if conclusion is None:
        return cleaned
    prefix = cleaned[: conclusion.start()].rstrip()
    return f"{prefix} Therefore, the answer is ({str(final_answer).upper()}) {option_text}.".strip()


@dataclass(frozen=True)
class ParsedMcqOutput:
    visible_text: str
    rationale: str | None
    rationale_only: str | None
    rationale_query: str | None
    answer_conclusion: str | None
    final_answer: str | None
    rationale_span: tuple[int, int] | None
    rationale_only_span: tuple[int, int] | None
    answer_conclusion_span: tuple[int, int] | None
    rationale_query_normalized: bool
    parse_errors: list[str]


def parse_mcq_output(text: str, options: dict[str, Any] | None) -> ParsedMcqOutput:
    visible_text = strip_hidden_thinking(text)
    valid_options = {str(label).upper() for label in (options or {})}
    errors: list[str] = []

    rationale_match = re.search(r"(?im)^\s*rationale\s*:\s*", visible_text)
    legacy_final_matches = list(
        re.finditer(r"(?im)^\s*(?:final\s+answer|final_answer)\s*:\s*([A-Za-z])\s*[.)]?\s*$", visible_text)
    )
    legacy_final_match = legacy_final_matches[-1] if legacy_final_matches else None

    rationale = None
    rationale_only = None
    rationale_query = None
    answer_conclusion = None
    rationale_span = None
    rationale_only_span = None
    answer_conclusion_span = None
    if rationale_match is None:
        # The paper prompt does not require a cosmetic ``Rationale:`` marker.
        # Preserve any visible explanation here; the conclusion parser below
        # reports whether generation ended before the required answer sentence.
        if visible_text:
            rationale = visible_text
            rationale_span = (0, len(visible_text))
        else:
            errors.append("empty_generation")
    else:
        start = rationale_match.end()
        end = legacy_final_match.start() if legacy_final_match is not None else len(visible_text)
        raw_rationale = visible_text[start:end]
        candidate = raw_rationale.strip()
        if candidate:
            rationale = candidate
            leading_whitespace = len(raw_rationale) - len(raw_rationale.lstrip())
            rationale_start = start + leading_whitespace
            rationale_span = (rationale_start, rationale_start + len(candidate))
        else:
            errors.append("empty_rationale")

    final_answer = None
    conclusion = None
    rationale_query_normalized = False
    if rationale:
        strict_conclusion = re.search(
            r"(?is)\btherefore\s*,?\s+the\s+answer\s+is\s*\(\s*([A-Za-z])\s*\)"
            r"(?:\s*[.:]\s*|\s+).*?\s*$",
            rationale,
        )
        paper_conclusion = re.search(
            r"(?is)\btherefore\s*,?\s+the\s+answer\s+is\s*\(?\s*([A-Za-z])\s*\)?"
            r"(?:\s*[.:]\s*|\s+).*?\s*$",
            rationale,
        )
        natural_conclusions = list(
            re.finditer(
                r"(?is)(?:\btherefore\s*,?\s*)?(?:\*{1,2})?(?:the\s+)?(?:final\s+)?"
                r"(?:(?:correct|best)\s+)?(?:answer|option|choice)\s*(?:is|:)\s*"
                r"(?:option\s*)?[\(\[]?\s*([A-Za-z])\s*[\)\]]?"
                r"(?:\s*[.:)\-]\s*.*?|\s+.*?)?\s*$",
                rationale,
            )
        )
        terminal_letter = re.search(
            r"(?is)(?:^|\n)\s*(?:\*{1,2})?[\(\[]?([A-Za-z])[\)\]]?[.)]?"
            r"(?:\*{1,2})?\s*\Z",
            rationale,
        )
        terminal_option = re.search(
            r"(?is)(?:^|\n)\s*(?:\*{1,2})?[\(\[]?([A-Za-z])[\)\]]?[.)]?\s+[^\n]+?"
            r"(?:\*{1,2})?\s*\Z",
            rationale,
        )
        conclusion = (
            strict_conclusion
            or paper_conclusion
            or (natural_conclusions[-1] if natural_conclusions else None)
            or terminal_letter
            or terminal_option
        )
        if conclusion is None:
            errors.append("missing_paper_answer_conclusion")
        else:
            candidate_answer = conclusion.group(1).upper()
            if candidate_answer in valid_options:
                final_answer = candidate_answer
            else:
                errors.append("invalid_answer_option")

            if rationale_span is not None:
                rationale_start = rationale_span[0]
                raw_reasoning = rationale[: conclusion.start()]
                reasoning = raw_reasoning.strip()
                if reasoning:
                    leading_whitespace = len(raw_reasoning) - len(raw_reasoning.lstrip())
                    reasoning_start = rationale_start + leading_whitespace
                    rationale_only = reasoning
                    rationale_only_span = (reasoning_start, reasoning_start + len(reasoning))

                raw_conclusion = rationale[conclusion.start() : conclusion.end()]
                cleaned_conclusion = raw_conclusion.strip()
                if cleaned_conclusion:
                    leading_whitespace = len(raw_conclusion) - len(raw_conclusion.lstrip())
                    conclusion_start = rationale_start + conclusion.start() + leading_whitespace
                    answer_conclusion = cleaned_conclusion
                    answer_conclusion_span = (
                        conclusion_start,
                        conclusion_start + len(cleaned_conclusion),
                    )

    if legacy_final_match is not None:
        legacy_answer = legacy_final_match.group(1).upper()
        if legacy_answer not in valid_options:
            errors.append("invalid_legacy_final_answer_option")
        elif final_answer is not None and legacy_answer != final_answer:
            errors.append("legacy_final_answer_mismatch")

    if rationale:
        rationale_query = canonicalize_rationale_query(rationale, final_answer, options)
        rationale_query_normalized = rationale_query != clean_text(rationale)

    return ParsedMcqOutput(
        visible_text=visible_text,
        rationale=rationale,
        rationale_only=rationale_only,
        rationale_query=rationale_query,
        answer_conclusion=answer_conclusion,
        final_answer=final_answer,
        rationale_span=rationale_span,
        rationale_only_span=rationale_only_span,
        answer_conclusion_span=answer_conclusion_span,
        rationale_query_normalized=rationale_query_normalized,
        parse_errors=errors,
    )


def parse_paper_exact_mcq_output(text: str, options: dict[str, Any] | None) -> ParsedMcqOutput:
    """Conservatively identify the final choice in a free paper-prompt response.

    RAG2's published prompt asks for a final option but prescribes no marker.
    We never rewrite the model response and accept a choice only when a
    terminal decision cue identifies one option.  The ordinary parser handles
    common ``answer is (A)`` forms first; this fallback additionally accepts
    a decision sentence such as ``Answer: ... B. ...`` or a unique option text
    stated after a conclusion cue.  Ambiguous, negated, and absent answers
    stay unparsed.
    """
    def has_multiple_explicit_labels(value: str) -> bool:
        """Return true only for an actual ``answer is A and B`` declaration.

        This intentionally does *not* look for a later ordinary ``and X`` in
        an answer sentence.  The previous broad expression did that and
        rejected valid completions such as ``The answer is C. The physician
        should ... and ...``.
        """
        return re.search(
            r"(?is)\b(?:the\s+)?(?:final\s+)?(?:correct|best)?\s*"
            r"(?:answer|option|choice)\b\s*(?:is\s*:?|:|-)?\s*(?:the\s+)?"
            r"(?:correct\s+)?(?:option\s*)?[\(\[]?\s*[A-Za-z]\s*[\)\]]?"
            r"\s*(?:and|or|,)\s*(?:option\s*)?[\(\[]?\s*[A-Za-z]"
            r"(?=\s*(?:[\)\].,:;]|$))",
            value,
        ) is not None

    parsed = parse_mcq_output(text, options)
    if parsed.final_answer is not None:
        matching_visible_text = re.sub(r"(?:\*{1,3}|_{1,3}|`)", "", parsed.visible_text)
        if not has_multiple_explicit_labels(matching_visible_text):
            return parsed
        return replace(
            parsed,
            final_answer=None,
            answer_conclusion=None,
            answer_conclusion_span=None,
            rationale_only=None,
            rationale_only_span=None,
            parse_errors=[*parsed.parse_errors, "ambiguous_multiple_final_answers"],
        )

    visible_text = parsed.visible_text
    valid_options = {str(label).upper(): clean_text(value) for label, value in (options or {}).items()}
    if not visible_text or not valid_options:
        return parsed
    if has_multiple_explicit_labels(re.sub(r"(?:\*{1,3}|_{1,3}|`)", "", visible_text)):
        return replace(
            parsed,
            final_answer=None,
            answer_conclusion=None,
            answer_conclusion_span=None,
            rationale_only=None,
            rationale_only_span=None,
            parse_errors=[*parsed.parse_errors, "ambiguous_multiple_final_answers"],
        )

    nonempty_lines = list(re.finditer(r"(?m)^.*\S.*$", visible_text))
    if not nonempty_lines:
        return parsed

    # Paper-exact responses freely use Markdown (for example
    # ``**Final answer:** C. ...``) and sometimes add a one-line explanation
    # *after* the explicit answer.  Match only the final short tail of the
    # response, strip presentation-only Markdown for matching, and never
    # modify the stored visible response.
    # A free response may state its choice and then give a short last
    # explanation.  Twelve logical lines still keeps extraction at the end of
    # the response, while covering that common layout.
    tail_lines = nonempty_lines[-12:]
    candidates: list[tuple[str, int, int, str]] = []

    explicit_patterns = [
        r"(?is)(?:therefore\s*,?\s*)?(?:the\s+)?(?:final\s+)?(?:correct|best)?\s*"
        r"(?:answer|option|choice)\s*(?:is\s*:?|:|-)\s*(?:the\s+)?(?:correct\s+)?"
        r"(?:option\s*)?[\(\[]?\s*([A-Za-z])(?![A-Za-z])\s*[\)\]]?",
        r"(?is)\b(?:final\s+answer\s*:\s*)?(?:the\s+)?(?:correct|best)\s+answer\s*"
        r"(?:is\s*:?)?\s*(?:the\s+)?(?:option\s*)?[\(\[]?\s*([A-Za-z])(?![A-Za-z])\s*[\)\]]?",
        r"(?is)\b(?:most\s+(?:likely|appropriate|probable|important|difficult|common)|best|correct|recommended)\b.*?"
        r"\b(?:is|are|would\s+be|is\s+to|should\s+be)\b\s*(?:to\s+)?"
        r"(?:the\s+)?(?:option\s*)?[\(\[]?\s*([A-Za-z])(?=\s*[\)\].,:;]|\s*$)\s*[\)\]]?",
        r"(?is)\b(?:most\s+(?:likely|appropriate|probable|important|difficult|common)|best|correct|recommended)\b.{0,240}?"
        r"[\(\[]\s*([A-Za-z])\s*[\)\]]",
        r"(?is)\b(?:most\s+(?:likely|appropriate|probable|important|difficult|common)|best|correct|recommended)\b"
        r".{0,360}?[\(\[]\s*(?:option|choice)\s*([A-Za-z])\s*[\)\]]",
        r"(?is)\b(?:this|that|it)\s+(?:is|would\s+be|corresponds\s+to|represents)\s*"
        r"(?:the\s+)?(?:correct\s+)?(?:option\s*)?[\(\[]?\s*([A-Za-z])"
        r"(?=\s*[\)\].,:;]|\s*$)\s*[\)\]]?",
        r"(?is)\b(?:i\s+(?:would\s+)?choose|i\s+select|would\s+choose|"
        r"choose|select|pick)\s+(?:the\s+)?(?:(?:option|choice)\s*)?[\(\[]?\s*([A-Za-z])"
        r"(?=\s*(?:[\(\[]|[\)\].,:;]|\b(?:as|because|which)\b|$))\s*[\)\]]?",
    ]
    # ``Options:`` is a section heading, not a decision cue.  In particular,
    # do not use the word "option" by itself here: it otherwise turns every
    # following ``A. ...`` / ``B. ...`` listing into an answer candidate.
    selection_cue = re.compile(
        r"(?i)\b(?:final|correct|best|answer|choice|conclusion|conclude|therefore|"
        r"most\s+(?:appropriate|likely|probable|important|difficult|common)|i\s+(?:would\s+)?choose|"
        r"i\s+select|would\s+choose|\b(?:choose|select|pick)\s+(?:the\s+)?(?:option|choice)|"
        r"treatment\s+of\s+choice|next\s+step|first\s+investigation|affects)\b"
    )
    bare_option = re.compile(
        r"(?is)^\s*(?:(?:option|choice)\s*)?[\(\[]?\s*([A-Za-z])\s*[\)\]]?"
        r"(?:\s*[.):\-]\s+\S|\s*$)"
    )
    selection_lead = re.compile(
        r"(?is)(?:\bbased\s+on\b|\btherefore\b|\bthus\b|\bhence\b|\bconclusion\b|"
        r"\b(?:best|correct|most\s+(?:likely|appropriate|probable|important|difficult|common))\b|"
        r"\b(?:next\s+step|treatment\s+of\s+choice|method\s+of\s+choice)\b)"
        r".{0,260}(?:\bis\b|\bare\b|:)\s*$"
    )
    trailing_option_label = re.compile(
        r"(?is)^.*?[\(\[]\s*(?:option|choice)?\s*([A-Za-z])\s*[\)\]]\s*[.!?]*\s*$"
    )

    def markdown_for_matching(value: str) -> str:
        # Only presentation markers are removed for matching; output text and
        # retrieval query remain exactly as generated.
        return re.sub(r"(?:\*{1,3}|_{1,3}|`)", "", value)

    # These phrases explicitly decline to select a valid MCQ option.  They
    # must not be converted into the first letter appearing in a parenthetic
    # list such as ``None of the above (A, B, C, or D)``.
    no_single_option = re.compile(
        r"(?is)\b(?:none\s+of\s+(?:the\s+)?(?:above|options?|choices?)|"
        r"not\s+(?:among|one\s+of)\s+(?:the\s+)?(?:options?|choices?)|"
        r"cannot\s+(?:choose|select|determine|provide|answer)|"
        r"(?:unable|impossible)\s+to\s+(?:choose|select|determine|provide|answer)|"
        r"will\s+not\s+(?:choose|select|provide)|"
        r"no\s+(?:option|answer|choice)\s+(?:is\s+)?(?:provided|given|correct|supported|mentioned))\b"
    )
    negated_decision = re.compile(
        r"(?is)\b(?:is|are|was|were)\s+not\s+(?:the\s+)?"
        r"(?:correct|best|likely|appropriate|supported)|"
        r"\bbut\s+(?:it\s+is\s+)?not\s+(?:correct|supported)\b"
    )

    for tail_index, line in enumerate(tail_lines):
        line_text = line.group(0).strip()
        line_start = line.start() + (len(line.group(0)) - len(line.group(0).lstrip()))
        matching_text = markdown_for_matching(line_text)
        previous_line = (
            markdown_for_matching(tail_lines[tail_index - 1].group(0)) if tail_index else ""
        )
        line_candidates: list[str] = []
        for pattern in explicit_patterns:
            for match in re.finditer(pattern, matching_text):
                label = match.group(1).upper()
                if label in valid_options:
                    line_candidates.append(label)

        # Accept ``A. option text`` only when an answer/conclusion cue appears
        # on this line or immediately before it. This avoids mistaking ordinary
        # option-by-option discussion for the final selection.
        bare_match = bare_option.match(matching_text)
        if bare_match is not None:
            label = bare_match.group(1).upper()
            previous_tail = " ".join(
                markdown_for_matching(previous.group(0)) for previous in tail_lines[max(0, tail_index - 2) : tail_index]
            )
            emphasized_tail_choice = (
                tail_index >= len(tail_lines) - 3
                and bool(
                    re.match(
                        r"^\s*(?:\*{1,3}|`)(?:(?:option|choice)\s*)?[\(\[]?[A-Za-z]",
                        line_text,
                        flags=re.IGNORECASE,
                    )
                )
            )
            if label in valid_options and (
                selection_cue.search(matching_text)
                or selection_cue.search(previous_tail)
                or selection_lead.search(previous_line)
                or emphasized_tail_choice
            ):
                line_candidates.append(label)

        # Free-form explanations often state the option text first and put the
        # label at its end, e.g. ``Reversible hydrocolloid (C)``.  It is safe
        # only in a conclusion-bearing line or immediately after one.
        trailing_label = trailing_option_label.match(matching_text)
        if trailing_label is not None:
            label = trailing_label.group(1).upper()
            if label in valid_options and (
                selection_cue.search(matching_text)
                or selection_lead.search(previous_line if tail_index else "")
                or (
                    tail_index >= len(tail_lines) - 3
                    and bool(re.match(r"^\s*(?:\*{1,3}|`)", line_text))
                )
            ):
                line_candidates.append(label)

        # Some generations give only the exact answer text after an explicit
        # final-answer marker.  Do not use topic overlap: accept only the
        # option text of a unique available choice on a cue-bearing line.
        compact_line = clean_text(matching_text).casefold().rstrip(".?!;:,)]}\"'")
        # An explicit option letter is more informative than the option text.
        # In particular, some MCQs contain duplicate option strings (e.g., B
        # and D have the same wording).  Adding both text matches after a
        # clear ``final answer is D`` declaration would manufacture an
        # ambiguity that is not present in the generated response.
        if selection_cue.search(matching_text) and not line_candidates:
            text_matches: list[tuple[str, int]] = []
            for label, option_text in valid_options.items():
                compact_option = clean_text(option_text).casefold().rstrip(".?!;:,)]}\"'")
                if compact_option and compact_line.endswith(compact_option):
                    text_matches.append((label, len(compact_option)))
            # Options can be nested strings (for example ``Retrograde`` and
            # ``Antegrade and retrograde``).  The longest exact tail is the
            # stated option; accepting both would incorrectly make a clear
            # answer look ambiguous.
            if text_matches:
                longest = max(length for _, length in text_matches)
                line_candidates.extend(label for label, length in text_matches if length == longest)

        # A response such as "the answer is A and B" is explicitly
        # non-single-choice and must remain unparsed.
        if has_multiple_explicit_labels(matching_text):
            line_candidates = []

        for label in set(line_candidates):
            candidates.append((label, line_start, line.start() + len(line.group(0)), "explicit_answer_tail"))

    # A free paper-prompt response may identify a choice and then continue its
    # explanation on the same line. Restrict this supplemental pass to the
    # final sentence fragments and require a decision cue, so ordinary
    # option-by-option discussion cannot become an answer accidentally.
    tail_start = nonempty_lines[max(0, len(nonempty_lines) - 16)].start()
    tail_text = visible_text[tail_start:]
    sentence_matches = list(re.finditer(r"(?s)[^.!?]+(?:[.!?]+(?=\s|$)|$)", tail_text))
    loose_label_patterns = [
        re.compile(
            r"(?is)\b(?:final\s+answer|correct\s+answer|answer|conclusion)\b\s*:?.{0,420}?"
            r"(?:\b(?:is|are|should\s+be|would\s+be|due\s+to|affects)\b\s*|:\s*)"
            r"(?:the\s+)?(?:correct\s+)?(?:option\s*)?[\(\[]?\s*([A-Za-z])(?=\s*[\)\].,:;])"
        ),
        re.compile(
            r"(?is)\b(?:therefore|thus|hence|conclude|treatment\s+of\s+choice|next\s+step|"
            r"first\s+investigation|most\s+(?:relevant|likely|appropriate|common)|best\s+(?:answer|option)|"
            r"affects)\b.{0,360}?"
            r"(?:\b(?:is|are|should\s+be|would\s+be|due\s+to|affects)\b\s*|:\s*)"
            r"(?:the\s+)?(?:correct\s+)?(?:option\s*)?[\(\[]?\s*([A-Za-z])(?=\s*[\)\].,:;])"
        ),
    ]
    for sentence in sentence_matches[-8:]:
        sentence_text = sentence.group(0).strip()
        if not sentence_text:
            continue
        matching_sentence = markdown_for_matching(sentence_text)
        if no_single_option.search(matching_sentence) or negated_decision.search(matching_sentence):
            continue
        sentence_start = tail_start + sentence.start() + (len(sentence.group(0)) - len(sentence.group(0).lstrip()))
        sentence_end = tail_start + sentence.end()
        sentence_candidates: list[str] = []
        for pattern in loose_label_patterns:
            for match in pattern.finditer(matching_sentence):
                if no_single_option.search(match.group(0)) or negated_decision.search(match.group(0)):
                    continue
                label = match.group(1).upper()
                if label in valid_options:
                    sentence_candidates.append(label)

        # Accept a unique full option text only inside a conclusion sentence.
        # For example: "the most common material is amalgam, because ...".
        # As above, do not let duplicate option wording override an explicit
        # terminal option label in the same sentence.
        if selection_cue.search(matching_sentence) and not sentence_candidates:
            compact_sentence = clean_text(matching_sentence).casefold()
            text_matches: list[tuple[str, int]] = []
            for label, option_text in valid_options.items():
                compact_option = clean_text(option_text).casefold()
                if len(compact_option) >= 4 and compact_option in compact_sentence:
                    text_matches.append((label, len(compact_option)))
            if text_matches:
                longest = max(length for _, length in text_matches)
                sentence_candidates.extend(label for label, length in text_matches if length == longest)

        if has_multiple_explicit_labels(matching_sentence):
            sentence_candidates = []
        for label in set(sentence_candidates):
            candidates.append((label, sentence_start, sentence_end, "terminal_decision_sentence"))

    if not candidates:
        return parsed
    # Prefer the final declaration, rather than rejecting a response merely
    # because its preceding conclusion mentioned a different option text.
    # This is common in free responses: e.g., "likely biphasic or triphasic"
    # followed by a definitive "Final answer: B. Biphasic".  A genuinely
    # multi-choice declaration has already been rejected above.  If a single
    # final line itself names incompatible labels, keep it unparsed.
    last_start = max(candidate[1] for candidate in candidates)
    final_candidates = [candidate for candidate in candidates if candidate[1] == last_start]
    labels = {candidate[0] for candidate in final_candidates}
    if len(labels) != 1:
        return parsed
    label = next(iter(labels))
    _, conclusion_start, conclusion_end, _ = final_candidates[-1]
    raw_reasoning = visible_text[:conclusion_start]
    rationale_only = raw_reasoning.strip() or None
    leading = len(raw_reasoning) - len(raw_reasoning.lstrip())
    rationale_only_span = (
        (leading, leading + len(rationale_only)) if rationale_only else None
    )
    answer_conclusion = visible_text[conclusion_start:conclusion_end].strip()
    conclusion_leading = len(visible_text[conclusion_start:conclusion_end]) - len(
        visible_text[conclusion_start:conclusion_end].lstrip()
    )
    answer_conclusion_span = (
        conclusion_start + conclusion_leading,
        conclusion_start + conclusion_leading + len(answer_conclusion),
    )
    errors = [error for error in parsed.parse_errors if error != "missing_paper_answer_conclusion"]
    return ParsedMcqOutput(
        visible_text=visible_text,
        rationale=visible_text,
        rationale_only=rationale_only,
        rationale_query=clean_text(visible_text),
        answer_conclusion=answer_conclusion,
        final_answer=label,
        rationale_span=(0, len(visible_text)),
        rationale_only_span=rationale_only_span,
        answer_conclusion_span=answer_conclusion_span,
        rationale_query_normalized=False,
        parse_errors=errors,
    )


def parse_paper_exact_terminal_mcq_output(
    text: str,
    options: dict[str, Any] | None,
) -> ParsedMcqOutput:
    """Parse only an exact canonical terminal line at the end of the response."""
    visible_text = strip_hidden_thinking(text)
    normalized = {str(label).upper(): clean_text(value) for label, value in (options or {}).items()}
    matches = [
        (label, paper_exact_terminal_line(normalized, label))
        for label in sorted(normalized)
        if visible_text.endswith(paper_exact_terminal_line(normalized, label))
    ]
    if len(matches) != 1:
        errors = ["missing_exact_terminal_answer"] if not matches else ["ambiguous_exact_terminal_answer"]
        return ParsedMcqOutput(
            visible_text=visible_text,
            rationale=visible_text or None,
            rationale_only=None,
            rationale_query=visible_text or None,
            answer_conclusion=None,
            final_answer=None,
            rationale_span=(0, len(visible_text)) if visible_text else None,
            rationale_only_span=None,
            answer_conclusion_span=None,
            rationale_query_normalized=False,
            parse_errors=errors,
        )

    label, terminal = matches[0]
    conclusion_start = len(visible_text) - len(terminal)
    # The terminal contract requires a separate last line.  Reject a suffix
    # embedded in the reasoning paragraph even if its wording happens to match.
    if conclusion_start > 0 and visible_text[conclusion_start - 1] != "\n":
        return ParsedMcqOutput(
            visible_text=visible_text,
            rationale=visible_text,
            rationale_only=None,
            rationale_query=visible_text,
            answer_conclusion=None,
            final_answer=None,
            rationale_span=(0, len(visible_text)),
            rationale_only_span=None,
            answer_conclusion_span=None,
            rationale_query_normalized=False,
            parse_errors=["terminal_answer_not_on_separate_line"],
        )
    rationale_only = visible_text[: max(0, conclusion_start - 1)].rstrip() or None
    rationale_only_span = (0, len(rationale_only)) if rationale_only else None
    return ParsedMcqOutput(
        visible_text=visible_text,
        rationale=visible_text,
        rationale_only=rationale_only,
        rationale_query=visible_text,
        answer_conclusion=terminal,
        final_answer=label,
        rationale_span=(0, len(visible_text)),
        rationale_only_span=rationale_only_span,
        answer_conclusion_span=(conclusion_start, len(visible_text)),
        rationale_query_normalized=False,
        parse_errors=[],
    )


def parse_mcq_output_for_prompt_profile(
    text: str,
    options: dict[str, Any] | None,
    prompt_profile: str,
) -> ParsedMcqOutput:
    """Parse a generated MCQ response without changing its visible wording.

    ``paper_exact`` has no terminal-string contract in the published prompt, so
    it needs the conservative final-line extractor above.  All other profiles
    retain the legacy parser and its explicit-answer contract.
    """
    if prompt_profile == "paper_exact":
        return parse_paper_exact_mcq_output(text, options)
    if prompt_profile == "paper_exact_terminal":
        return parse_paper_exact_terminal_mcq_output(text, options)
    return parse_mcq_output(text, options)


def gold_answers(row: dict[str, Any]) -> set[str]:
    valid_options = set(normalized_options(row))
    values = row.get("answers")
    if not isinstance(values, list):
        values = [row.get("answer")]
    return {str(value).upper() for value in values if str(value).upper() in valid_options}


def is_correct(row: dict[str, Any], prediction: str | None) -> bool:
    return prediction is not None and prediction in gold_answers(row)

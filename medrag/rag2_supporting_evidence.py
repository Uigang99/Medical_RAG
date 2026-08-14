from __future__ import annotations

import json
import re
from typing import Any

from .rag2_mcq import clean_text, format_question, render_paper_document_view


SUPPORTING_SENTENCE_PROMPT_VERSION = "rag2_llama3_supporting_sentence_ids_v2"
DOCUMENT_SENTENCE_SEGMENTATION_VERSION = "rag2_document_sentence_offsets_v2"
INLINE_SUPPORTING_TRACE_PROMPT_VERSION = "rag2_llama3_inline_rationale_answer_support_v6"
INLINE_SUPPORTING_TRACE_OUTPUT_VERSION = "rag2_inline_support_sentence_ids_json_v6"

_SUPPORT_JSON_HEADER = "SUPPORTING_SENTENCE_JSON:"

# This regular language is intentionally limited to the terminal machine
# section.  ``(.|\n)*`` leaves the rationale and answer entirely free-form,
# while the suffix guarantees one newline-delimited JSON object, one to four
# canonical sentence IDs (or NONE), and no trailing prose.  It is compatible
# with vLLM's xgrammar backend, which does not support escaped literal braces
# or bounded ``{m,n}`` repetitions in regex mode.
INLINE_SUPPORTING_TRACE_TERMINAL_REGEX = (
    r'(.|\n)*\nSUPPORTING_SENTENCE_JSON: [{}]"supporting_sentence_ids":('
    r'"NONE"|'
    r'\["S[0-9][0-9][0-9]"\]|'
    r'\["S[0-9][0-9][0-9]","S[0-9][0-9][0-9]"\]|'
    r'\["S[0-9][0-9][0-9]","S[0-9][0-9][0-9]","S[0-9][0-9][0-9]"\]|'
    r'\["S[0-9][0-9][0-9]","S[0-9][0-9][0-9]","S[0-9][0-9][0-9]","S[0-9][0-9][0-9]"\]'
    r')[{}]'
)


def render_document_view(document: dict[str, Any], max_doc_chars: int = 2600) -> str:
    """Return the exact document text appended by the paper-style prompt."""
    return render_paper_document_view(document, max_doc_chars)


def segment_document_view(
    document_view: str,
    *,
    document_title: str | None = None,
) -> list[dict[str, Any]]:
    """Split a rendered document into stable sentence IDs with source offsets.

    The splitter is deliberately lightweight and deterministic. It preserves
    exact substrings from the text shown to the model, which matters more for
    later sentence-removal interventions than linguistic perfection. A title
    remains visible to the model but is never selectable evidence: a title is
    metadata, not a sentence that can materially support a response.
    """
    title = clean_text(document_title)
    content_start = 0
    if title and document_view.startswith(title):
        if len(document_view) == len(title):
            return []
        if document_view[len(title) :].startswith("\n"):
            content_start = len(title) + 1
    elif document_title is None and "\n" in document_view:
        # ``render_paper_document_view`` uses its only newline to separate a
        # title from body text. Preserve backwards-compatible direct calls
        # while retaining the no-title-ID contract.
        content_start = document_view.index("\n") + 1

    boundaries = {content_start, len(document_view)}
    for match in re.finditer(r"\n+", document_view):
        if match.end() > content_start:
            boundaries.update((max(content_start, match.start()), match.end()))
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|(?<=;)\s+(?=[A-Z0-9(\[])", document_view):
        if match.end() > content_start:
            boundaries.update((max(content_start, match.start()), match.end()))

    ordered = sorted(boundaries)
    sentences: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        raw = document_view[left:right]
        stripped = raw.strip()
        if not stripped:
            continue
        start = left + len(raw) - len(raw.lstrip())
        end = right - (len(raw) - len(raw.rstrip()))
        sentences.append(
            {
                "sentence_id": f"S{len(sentences) + 1:03d}",
                "text": document_view[start:end],
                "char_start": start,
                "char_end": end,
            }
        )
    return sentences


def render_tagged_document_view(
    document_view: str,
    sentences: list[dict[str, Any]],
    *,
    document_title: str | None = None,
) -> str:
    """Render the answer-time document with stable IDs on selectable body text.

    The title remains in the same visible form as the original document view,
    but it deliberately has no ID.  This makes sentence attribution possible
    during the *same* response generation without turning a title into evidence.
    """
    title = clean_text(document_title)
    tagged_body = "\n".join(
        f"[{sentence['sentence_id']}] {sentence['text']}" for sentence in sentences
    )
    if title:
        return "\n".join(part for part in (title, tagged_body) if part)
    return tagged_body


def build_inline_supporting_trace_messages(
    row: dict[str, Any],
    document_view: str,
    sentences: list[dict[str, Any]],
    *,
    document_title: str | None = None,
    format_retry: bool = False,
    format_retry_round: int = 0,
    compact_retry: bool = False,
) -> list[dict[str, str]]:
    """Build one w/RAG generation that emits rationale, answer, and attribution.

    Supporting evidence is generated after the answer as part of the same
    autoregressive trace.  It is an attribution of the already expressed
    rationale, not a second answer-generation pass.
    """
    tagged_document = render_tagged_document_view(
        document_view,
        sentences,
        document_title=document_title,
    )
    retry_instruction = ""
    if format_retry:
        retry_variants = (
            "The previous response was rejected because its final evidence line was not machine-readable. "
            "Regenerate the complete response and obey the final-line contract exactly. ",
            "Schema repair is required. Regenerate the complete response; before finishing, verify that its last "
            "characters are the closing brace of the required evidence JSON. ",
            "This is the final format-repair attempt. Regenerate the full response and do not stop after the answer: "
            "the terminal evidence JSON line is mandatory. ",
        )
        retry_instruction = retry_variants[min(max(format_retry_round, 1), len(retry_variants)) - 1]
    compact_instruction = (
        "The previous response exceeded the output limit. Keep the rationale to only the decisive medical facts and "
        "use at most two supporting sentences unless more are indispensable. "
        if compact_retry
        else ""
    )
    instruction = (
        "The following are multiple choice questions about medical knowledge. Solve the question and provide a "
        "concise medical rationale followed by one answer from the listed options. "
        "Start directly with the rationale; do not use headings, bullets, or document citations in the rationale. "
        "End the rationale with a separate answer line in exactly this form: "
        "'Therefore, the answer is (<OPTION LETTER>) <EXACT OPTION TEXT>.'. "
        "Immediately after that answer line, output exactly one final line beginning with "
        f"'{_SUPPORT_JSON_HEADER}'. The rest of that line must be one valid JSON object in exactly this form: "
        '{"supporting_sentence_ids":["S001","S002"]}. Stop immediately after the final JSON brace; do not use XML '
        "tags, markdown fences, headings, or any text after the JSON.\n\n"
        "Supporting Sentence (S) means a document sentence whose medical information you identified as being "
        "meaningfully reflected in the reasoning used to infer the answer. Select sentences containing a medical "
        "fact, mechanism, diagnostic criterion, treatment rule, exclusion, or other clinical premise that is relevant "
        "to the answer or rationale. Do not select a sentence for topic, word, disease-name, entity, or option overlap "
        "alone. This is a recall-oriented attribution: when a body sentence contains a plausible medical premise used "
        "in the reasoning, include it rather than omitting it because the linkage is not perfectly explicit. Select all "
        "such key premises, up to four body-sentence IDs; do not stop at one minimal citation when additional factual "
        "premises are used. A document heading or title is not itself evidence: inspect the following body sentences and "
        "select their factual medical content instead. Do not select a title or question restatement. Output NONE "
        "only when the document has no body sentence with medically meaningful information relevant to the reasoning. "
        "For NONE, output exactly {\"supporting_sentence_ids\":\"NONE\"}. The document title is metadata and cannot "
        "be selected. Do not solve the question again in the JSON."
    )
    terminal_reminder = (
        "\n\nFinal output reminder: write the rationale and exact answer line first. Then write one last line only: "
        f"{_SUPPORT_JSON_HEADER} {{\"supporting_sentence_ids\":[\"S001\"]}} "
        "(or the specified NONE object). End immediately after its final `}`."
    )
    content = (
        f"{retry_instruction}{compact_instruction}{instruction}\n\n"
        f"Question and options:\n{format_question(row)}\n\n"
        "Retrieved document (only bracketed body IDs are selectable):\n"
        f"{tagged_document or '(No selectable document body sentences are available.)'}"
        f"{terminal_reminder}"
    )
    return [{"role": "user", "content": content}]


def split_inline_supporting_trace(raw_text: str) -> dict[str, Any]:
    """Split a same-trace rationale/answer response from its terminal S section."""
    visible = str(raw_text or "").strip()
    # The instruction requests a separate line, but accepting a unique header
    # immediately after the answer avoids discarding a fully recoverable trace
    # when a decoder omits that single newline.  The raw text is retained.
    pattern = re.compile(rf"(?i){re.escape(_SUPPORT_JSON_HEADER)}\s*")
    matches = list(pattern.finditer(visible))
    issues: list[str] = []
    if len(matches) != 1:
        issues.append("missing_or_multiple_supporting_evidence_json_headers")
        return {
            "raw_generation": visible,
            "response_text": visible,
            "support_raw_generation": "",
            "support_span": None,
            "quality_issues": issues,
        }
    match = matches[0]
    response_text = visible[: match.start()].rstrip()
    if not response_text:
        issues.append("missing_rationale_and_answer_before_support_section")
    return {
        "raw_generation": visible,
        "response_text": response_text,
        "support_raw_generation": visible[match.end() :].strip(),
        "support_span": (match.end(), len(visible)),
        "quality_issues": issues,
    }


def build_supporting_sentence_messages(
    row: dict[str, Any],
    document_view: str,
    sentences: list[dict[str, Any]],
    free_generation: str,
    *,
    document_title: str | None = None,
    json_only_retry: bool = False,
) -> list[dict[str, str]]:
    tagged_document = "\n".join(
        f"[{sentence['sentence_id']}] {sentence['text']}" for sentence in sentences
    )
    title = clean_text(document_title)
    title_block = f"Document title (metadata only; it has no sentence ID):\n{title}\n\n" if title else ""
    if json_only_retry:
        instruction = (
            "Repair the sentence-ID selection. Return ONLY one valid JSON object and nothing else: "
            '{"supporting_sentence_ids":["S001","S002"]}. Use only the exact displayed IDs. '
            "Never select the untagged document title. If no body sentence materially supports the generated "
            "response, return {\"supporting_sentence_ids\":[]}."
        )
    else:
        instruction = (
            "Identify the sentence or sentences from the provided document whose information is explicitly used in, "
            "or materially supports, the generated medical reasoning or final answer. Select only sentences that "
            "contribute information to the response; topical similarity alone is not sufficient. The document title "
            "is metadata and has no sentence ID, so it must never be selected. Do not solve the question again and "
            "do not add an explanation. If no document body sentence contributes, return an empty list. Return exactly "
            "one JSON object in this form: {\"supporting_sentence_ids\":[\"S001\",\"S002\"]}."
        )
    content = (
        f"{instruction}\n\n"
        f"Question and options:\n{format_question(row)}\n\n"
        f"Generated response:\n{free_generation}\n\n"
        f"{title_block}"
        f"Document body sentences shown during generation:\n{tagged_document or '(No selectable body sentences.)'}"
    )
    return [{"role": "user", "content": content}]


def parse_supporting_sentence_output(
    raw_text: str,
    sentences: list[dict[str, Any]],
    *,
    require_json_only: bool = False,
) -> dict[str, Any]:
    """Parse a terminal supporting-sentence ID list without semantic post-processing."""
    visible = str(raw_text or "").strip()
    allowed = {sentence["sentence_id"]: sentence for sentence in sentences}
    issues: list[str] = []
    warnings: list[str] = []
    mode = "strict_json"
    values: Any = None
    original_values: list[Any] | None = None

    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", visible, flags=re.IGNORECASE).strip()
    if require_json_only and visible.startswith("```"):
        issues.append("supporting_evidence_markdown_fence_not_allowed")
    try:
        parsed = json.loads(candidate)
        values = parsed.get("supporting_sentence_ids") if isinstance(parsed, dict) else None
        if isinstance(values, str) and values.strip().upper() == "NONE":
            values = []
        elif isinstance(values, list) and len(values) == 1 and str(values[0]).strip().upper() == "NONE":
            values = []
        if not isinstance(values, list):
            issues.append("missing_supporting_sentence_ids_list")
            values = None
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                values = parsed.get("supporting_sentence_ids")
                if isinstance(values, str) and values.strip().upper() == "NONE":
                    values = []
                elif isinstance(values, list) and len(values) == 1 and str(values[0]).strip().upper() == "NONE":
                    values = []
                mode = "embedded_json"
                warnings.append("json_prefix_or_suffix_ignored")
                if not isinstance(values, list):
                    issues.append("missing_supporting_sentence_ids_list")
                    values = None
                break
        if values is None and not issues:
            issues.append("invalid_json")

    if values is None:
        mode = "id_regex_fallback"
        values = re.findall(r"\bS\d{3,}\b", visible.upper())
        if values:
            warnings.append("id_regex_fallback_for_audit_only")
        else:
            issues.append("unparseable_supporting_sentence_output")
    else:
        original_values = list(values)

    normalized: list[str] = []
    invalid: list[str] = []
    for value in values:
        raw_id = str(value).strip().upper()
        match = re.fullmatch(r"S0*(\d+)", raw_id)
        sentence_id = f"S{int(match.group(1)):03d}" if match and int(match.group(1)) > 0 else raw_id
        if sentence_id not in allowed:
            invalid.append(raw_id)
        elif sentence_id not in normalized:
            normalized.append(sentence_id)
    if invalid:
        issues.append("unknown_supporting_sentence_ids")

    # A non-JSON fallback is useful diagnostic information, but it does not
    # satisfy the storage contract. The caller will issue one support-only
    # JSON repair attempt without regenerating the answer or PPL scores.
    if mode == "id_regex_fallback":
        issues.append("supporting_sentence_json_required")
    if require_json_only and mode != "strict_json":
        issues.append("supporting_sentence_json_must_be_only_section_content")

    return {
        "raw_generation": visible,
        "parse_mode": mode,
        "sentence_ids": normalized,
        "sentences": [allowed[sentence_id] for sentence_id in normalized],
        "support_links": [],
        "document_use": "NONE" if not normalized else "SUPPORTED",
        "none_reason": None,
        "original_sentence_ids": original_values,
        "invalid_sentence_ids": invalid,
        "quality_pass": not issues,
        "retry_required": bool(issues),
        "quality_issues": issues,
        "parse_warnings": warnings,
        "document_sentence_count": len(sentences),
    }

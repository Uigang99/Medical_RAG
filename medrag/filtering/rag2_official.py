from __future__ import annotations

from typing import Any, Mapping


HELPFUL_TOKEN = "[HELPFUL]"
NOT_HELPFUL_TOKEN = "[NOT_HELPFUL]"
DISCARD_TOKEN = "[DISCARD]"
LABEL_TOKENS = (HELPFUL_TOKEN, NOT_HELPFUL_TOKEN)
LABEL_NAMES = ("helpful", "not helpful")
THREE_CLASS_LABEL_TOKENS = (HELPFUL_TOKEN, NOT_HELPFUL_TOKEN, DISCARD_TOKEN)
THREE_CLASS_LABEL_NAMES = ("helpful", "not helpful", "discard")
OFFICIAL_INSTRUCTION = "Given the following evidence, determine whether it helps answer the provided question."
RATIONALE_AWARE_INSTRUCTION = (
    "Given an initial rationale, the following evidence, and a question, determine whether "
    "the evidence helps answer the question."
)
ANSWER_AWARE_INSTRUCTION = (
    "Given an initial answer generated without retrieved evidence, the following evidence, "
    "and a question, determine whether the evidence helps answer the question."
)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def format_options(options: Mapping[str, Any] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{key}) {clean_text(options[key])}" for key in sorted(options))


def normalize_option_lines(options: str) -> str:
    lines: list[str] = []
    for raw_line in str(options or "").splitlines():
        line = raw_line.strip()
        if len(line) >= 3 and line[0].isalnum() and line[1:3] == ". ":
            line = f"{line[0]}) {line[3:]}"
        if line:
            lines.append(line)
    return "\n".join(lines)


def build_official_filter_input(question: str, evidence: str, options: str = "") -> str:
    question_block = clean_text(question)
    options_block = str(options or "").strip()
    if options_block:
        question_block = f"{question_block}\n{options_block}"
    return (
        f"{OFFICIAL_INSTRUCTION}\n\n"
        f"Evidence: {clean_text(evidence)}\n\n"
        f"Question: {question_block}"
    )


def build_rationale_aware_filter_input(base_input: str, no_rag_rationale: str) -> str:
    """Extend the released filter schema with the available no-RAG rationale.

    This is intentionally an ablation rather than the paper's filter input. The
    original evidence and question blocks are retained byte-for-byte after
    normalization so only the rationale availability changes.
    """

    normalized_input = convert_legacy_filter_input(base_input)
    if not normalized_input.startswith(OFFICIAL_INSTRUCTION):
        raise ValueError("Expected the normalized RAG2 evidence-question filter input.")
    suffix = normalized_input[len(OFFICIAL_INSTRUCTION) :].lstrip()
    rationale = clean_text(no_rag_rationale)
    if not rationale:
        raise ValueError("No-RAG rationale must be non-empty for rationale-aware filtering.")
    return (
        f"{RATIONALE_AWARE_INSTRUCTION}\n\n"
        f"Initial rationale generated without retrieved evidence: {rationale}\n\n"
        f"{suffix}"
    )


def build_answer_aware_filter_input(base_input: str, no_rag_answer: str) -> str:
    """Add only the target model's cached No-RAG answer to the official input.

    The answer is a model prediction available at deployment time.  Gold
    answers, answer correctness, confidence, margins, and rationale text are
    deliberately absent from this ablation.
    """

    normalized_input = convert_legacy_filter_input(base_input)
    if not normalized_input.startswith(OFFICIAL_INSTRUCTION):
        raise ValueError("Expected the normalized RAG2 evidence-question filter input.")
    suffix = normalized_input[len(OFFICIAL_INSTRUCTION) :].lstrip()
    answer = clean_text(no_rag_answer)
    if not answer:
        raise ValueError("No-RAG answer must be non-empty for answer-aware filtering.")
    return (
        f"{ANSWER_AWARE_INSTRUCTION}\n\n"
        f"Initial answer generated without retrieved evidence: {answer}\n\n"
        f"{suffix}"
    )


def convert_legacy_filter_input(text: str) -> str:
    """Convert this project's existing pseudo-label rows to the official RAG2 schema."""
    value = str(text or "")
    question_marker = "Question:\n"
    options_marker = "\n\nOptions:\n"
    evidence_marker = "\n\nRetrieved document:\n"
    answer_marker = "\n\nAnswer with exactly one label:"
    try:
        question_start = value.index(question_marker) + len(question_marker)
        options_start = value.index(options_marker, question_start)
        evidence_start = value.index(evidence_marker, options_start)
    except ValueError as exc:
        if value.startswith(OFFICIAL_INSTRUCTION):
            return value
        raise ValueError("Could not parse the legacy RAG2 filter input format.") from exc

    question = value[question_start:options_start]
    options = normalize_option_lines(value[options_start + len(options_marker) : evidence_start])
    document_start = evidence_start + len(evidence_marker)
    answer_start = value.find(answer_marker, document_start)
    evidence = value[document_start:] if answer_start < 0 else value[document_start:answer_start]
    return build_official_filter_input(question=question, options=options, evidence=evidence)


def extract_official_evidence(text: str) -> str:
    """Return the Evidence field from an official (or convertible legacy) input.

    This is used only for validation-time window-threshold calibration.  It
    deliberately preserves the same normalization boundary as the filter
    trainer: the caller receives the evidence value before it is split into
    sentence-context windows.
    """

    normalized = convert_legacy_filter_input(text)
    prefix = f"{OFFICIAL_INSTRUCTION}\n\nEvidence: "
    question_marker = "\n\nQuestion: "
    if not normalized.startswith(prefix):
        raise ValueError("official_filter_input_prefix_mismatch")
    question_index = normalized.find(question_marker, len(prefix))
    if question_index < 0:
        raise ValueError("official_filter_input_question_marker_missing")
    evidence = clean_text(normalized[len(prefix) : question_index])
    if not evidence:
        raise ValueError("official_filter_input_evidence_empty")
    return evidence


def replace_official_evidence(text: str, evidence: Any) -> str:
    """Replace only an official filter input's Evidence value.

    Window inference/calibration must keep every question and option character
    byte-for-byte identical to the training row.  Rebuilding the prompt from
    parsed question text would make that contract needlessly fragile.
    """

    normalized = convert_legacy_filter_input(text)
    prefix = f"{OFFICIAL_INSTRUCTION}\n\nEvidence: "
    question_marker = "\n\nQuestion: "
    if not normalized.startswith(prefix):
        raise ValueError("official_filter_input_prefix_mismatch")
    question_index = normalized.find(question_marker, len(prefix))
    if question_index < 0:
        raise ValueError("official_filter_input_question_marker_missing")
    replacement = clean_text(evidence)
    if not replacement:
        raise ValueError("official_filter_input_replacement_evidence_empty")
    return f"{prefix}{replacement}{normalized[question_index:]}"


def label_name(value: Any) -> str:
    normalized = clean_text(value).lower().replace("_", " ")
    normalized = normalized.strip("[]")
    if normalized == "helpful":
        return "helpful"
    if normalized in {"not helpful", "nothelpful", "unhelpful"}:
        return "not helpful"
    if normalized in {"discard", "abstain", "no decision"}:
        return "discard"
    raise ValueError(f"Unsupported RAG2 filter label: {value!r}")


def label_token(value: Any) -> str:
    name = label_name(value)
    if name == "helpful":
        return HELPFUL_TOKEN
    if name == "not helpful":
        return NOT_HELPFUL_TOKEN
    return DISCARD_TOKEN


def add_label_tokens(tokenizer: Any, model: Any) -> dict[str, int]:
    """Mirror the official repository's tokenizer.add_tokens + resize workflow."""
    # The public Flan-T5 checkpoint has untied encoder/decoder output embeddings.
    # Keep that architecture when Transformers resizes the vocabulary.
    model.config.tie_word_embeddings = False
    tokenizer.add_tokens(list(LABEL_TOKENS))
    model.resize_token_embeddings(len(tokenizer))
    token_ids = resolve_label_token_ids(tokenizer)
    model.config.rag2_filter_label_names = list(LABEL_NAMES)
    model.config.rag2_filter_label_tokens = list(LABEL_TOKENS)
    model.config.rag2_filter_input_format = "rag2_official_evidence_question_v1"
    model.config.rag2_filter_decision_rule = "first_decoder_step_two_token_softmax"
    return token_ids


def resolve_label_token_ids(tokenizer: Any) -> dict[str, int]:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    token_ids: dict[str, int] = {}
    for name, token in zip(LABEL_NAMES, LABEL_TOKENS):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or (unk_id is not None and ids[0] == unk_id):
            raise ValueError(
                f"RAG2 label {token!r} must exist as exactly one tokenizer token; got ids={ids}."
            )
        token_ids[name] = int(ids[0])
    return token_ids


def resolve_configured_label_token_ids(tokenizer: Any, config: Any) -> tuple[tuple[str, ...], dict[str, int]]:
    """Resolve the label space recorded by a trained filter checkpoint.

    Historical binary checkpoints do not carry ``rag2_filter_label_names``;
    they intentionally fall back to the released two-label contract.  New
    three-class checkpoints store both names and atomic tokens in config.json,
    which lets every downstream scorer treat Discard as an abstention result
    while retaining only Helpful documents.
    """

    configured_names = getattr(config, "rag2_filter_label_names", None)
    configured_tokens = getattr(config, "rag2_filter_label_tokens", None)
    if configured_names is None and configured_tokens is None:
        return LABEL_NAMES, resolve_label_token_ids(tokenizer)
    # Some historical checkpoints recorded only the token list.  Infer the
    # canonical ordered names from its length so this extension remains fully
    # backward compatible with those binary artifacts.
    if configured_names is None and isinstance(configured_tokens, (list, tuple)):
        configured_names = LABEL_NAMES if len(configured_tokens) == 2 else THREE_CLASS_LABEL_NAMES
    if configured_tokens is None and isinstance(configured_names, (list, tuple)):
        configured_tokens = LABEL_TOKENS if len(configured_names) == 2 else THREE_CLASS_LABEL_TOKENS
    if not isinstance(configured_names, (list, tuple)) or not isinstance(configured_tokens, (list, tuple)):
        raise ValueError("Filter config must provide a resolvable label-name/token contract.")
    names = tuple(label_name(value) for value in configured_names)
    tokens = tuple(str(value) for value in configured_tokens)
    if len(names) != len(tokens) or len(set(names)) != len(names):
        raise ValueError(f"Invalid configured filter label contract: names={names}, tokens={tokens}")
    if names not in {LABEL_NAMES, THREE_CLASS_LABEL_NAMES}:
        raise ValueError(f"Unsupported configured filter label order: {names}")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    token_ids: dict[str, int] = {}
    for name, token in zip(names, tokens):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or (unk_id is not None and ids[0] == unk_id):
            raise ValueError(f"RAG2 label {token!r} must resolve to one non-UNK token; got ids={ids}.")
        token_ids[name] = int(ids[0])
    return names, token_ids

from pathlib import Path

import pytest

from medrag.rag2_anchored_trace import (
    ANCHOR_NAMES,
    END_REASONING_MARKER,
    FINAL_ANSWER_PREFIX,
    build_anchored_user_prompt,
    canonical_response,
    encode_to_pre_choice,
    normalize_rationale,
    semantic_retrieval_queries,
)


ROW = {
    "question": "Which treatment is most appropriate?",
    "options": {"A": "Observation", "B": "Antibiotics", "C": "Intubation", "D": "Surgery"},
    "answer": "C",
}


def test_canonical_response_and_query_exclude_control_marker() -> None:
    rationale = "The patient has impending respiratory failure."
    response = canonical_response(rationale, "C", ROW["options"])
    assert response == (
        "Rationale:\n"
        "The patient has impending respiratory failure.\n"
        "### END OF REASONING ###\n"
        "Final answer: (C) Intubation"
    )
    queries = semantic_retrieval_queries(rationale, "C", ROW["options"])
    assert END_REASONING_MARKER not in queries["rationale_answer"]
    assert queries["rationale_answer"].endswith("Final answer: (C) Intubation")


def test_prompt_uses_raw_document_without_source_metadata() -> None:
    prompt = build_anchored_user_prompt(ROW, "Decisive evidence only.")
    assert "Documents:\nDecisive evidence only." in prompt
    assert "source:" not in prompt.lower()
    assert "rank:" not in prompt.lower()


def test_normalize_rationale_flags_embedded_terminal() -> None:
    text, flags = normalize_rationale(
        "Clinical reasoning.\nFinal answer: (C) Intubation"
    )
    assert text == "Clinical reasoning."
    assert flags == ["embedded_final_answer"]


def test_llama_anchor_token_order_and_suffix() -> None:
    transformers = pytest.importorskip("transformers")
    model_path = Path("/home/user/Uiheon/models/Llama-3-8B-Instruct")
    if not model_path.is_dir():
        pytest.skip("Local Llama tokenizer is unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    encoded = encode_to_pre_choice(
        tokenizer,
        ROW,
        "The document describes severe respiratory failure.",
        "The findings indicate that airway protection is required.",
    )
    indices = [encoded.anchor_indices[name] for name in ANCHOR_NAMES]
    assert indices == sorted(indices)
    assert len(set(indices)) == 3
    assert encoded.full_prefix_text.endswith(FINAL_ANSWER_PREFIX)
    assert encoded.anchor_token_text["pre_choice"]

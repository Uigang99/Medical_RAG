"""Deterministic sentence-context windows for attribution-window RAG² filters.

The attribution-window training data contains the highest-attribution sentence
and one neighbouring sentence on each side.  At benchmark inference time that
highest-attribution sentence is unavailable (it would require the answer that
we are trying to generate), so this module enumerates *all* possible centred
windows under the same sentence segmentation and context rule.

The evaluator later aggregates those window scores back to one document score.
Keeping window construction in one small shared module prevents the training
and evaluation contracts from drifting apart.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..rag2_supporting_evidence import (
    DOCUMENT_SENTENCE_SEGMENTATION_VERSION,
    segment_document_view,
)
from .rag2_official import clean_text


WINDOWING_VERSION = "rag2_sentence_context_sliding_window_v1"


def sentence_context_windows(
    evidence: Any,
    *,
    context_sentences: int = 1,
) -> list[dict[str, Any]]:
    """Return unique centred sentence windows from one filter evidence field.

    ``context_sentences=1`` exactly matches the Top-1-attribution-sentence
    plus preceding/following context used for the current training inputs.
    Evidence is cleaned before segmentation just as the training-input builder
    cleaned its stored representative window.  Titles are deliberately not
    treated as separate evidence metadata: callers pass ``text or title``,
    which matches the historical filter-data materialisation fallback.
    """

    if context_sentences < 0:
        raise ValueError("context_sentences must be non-negative")
    document_text = clean_text(evidence)
    if not document_text:
        return []
    sentences = segment_document_view(document_text, document_title="")
    if not sentences:
        return []

    windows: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    for centre_index in range(len(sentences)):
        start_index = max(0, centre_index - context_sentences)
        end_index = min(len(sentences), centre_index + context_sentences + 1)
        # A two-sentence document produces the same S001-S002 window when
        # either sentence is chosen as the centre.  Scoring it twice would
        # alter the multiple-instance false-positive rate for no information.
        span = (start_index, end_index)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        members = sentences[start_index:end_index]
        text = " ".join(str(sentence["text"]) for sentence in members)
        windows.append(
            {
                "window_id": f"{members[0]['sentence_id']}-{members[-1]['sentence_id']}",
                "centre_sentence_id": sentences[centre_index]["sentence_id"],
                "sentence_ids": [sentence["sentence_id"] for sentence in members],
                "sentence_count": len(members),
                "char_start": int(members[0]["char_start"]),
                "char_end": int(members[-1]["char_end"]),
                "text": text,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return windows


def windowing_contract(context_sentences: int) -> dict[str, Any]:
    """Serializable identity used by score caches and calibration artifacts."""

    return {
        "windowing_version": WINDOWING_VERSION,
        "sentence_segmentation_version": DOCUMENT_SENTENCE_SEGMENTATION_VERSION,
        "context_sentences": int(context_sentences),
        "deduplicate_identical_sentence_spans": True,
        "text_normalization": "rag2_official_clean_text",
    }

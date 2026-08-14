from __future__ import annotations

import re
from typing import Any


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"(?is)\[/?think\]", " ", text)
    return text


def _extract_after_final_marker(text: str) -> str:
    markers = list(
        re.finditer(
            r"(?is)(?:^|\n|\b)(?:final\s+(?:answer|response)|answer|response)\s*[:：\-]\s*",
            text,
        )
    )
    if not markers:
        return text
    return text[markers[-1].end() :]


def postprocess_generation_text(text: str, *, max_tokens: int = 256) -> str:
    """Convert raw model text into the answer string used for saving/evaluation."""
    value = _safe_str(text)
    if not value.strip():
        return ""

    value = _strip_thinking_blocks(value)
    value = _extract_after_final_marker(value)
    value = re.sub(r"`{3,}\s*[a-zA-Z0-9_+\-]*\s*", " ", value)
    value = value.replace("```", " ")

    raw_lines = [line.strip() for line in value.splitlines() if line.strip()]
    drop_patterns = [
        r"^</?think>$",
        r"^/(?:no_)?think$",
        r"^assistant\s*[:：]?$",
        r"^analysis\s*[:：]?$",
        r"^thinking\s+process\s*[:：]?$",
    ]

    lines: list[str] = []
    for line in raw_lines:
        if any(re.fullmatch(pattern, line, flags=re.IGNORECASE) for pattern in drop_patterns):
            continue
        lines.append(line)

    if lines:
        value = " ".join(lines)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""

    value = re.sub(r"^\s*(?:[-*]\s*)?(?:final\s+(?:answer|response)|answer|response)\s*[:：\-]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:the\s+answer\s+is)\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip("\"'`").strip()
    value = re.sub(r"\s+", " ", value).strip()

    tokens = value.split()
    if max_tokens > 0 and len(tokens) > max_tokens:
        value = " ".join(tokens[:max_tokens]).strip()
    return value

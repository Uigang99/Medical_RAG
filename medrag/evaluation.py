from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from .core import BenchmarkSample


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_mcq_choice(prediction: str, valid_choices: set[str]) -> str | None:
    head = prediction[:200].upper()
    patterns = [
        r"(?:ANSWER|OPTION|정답)\s*[:\-]?\s*([A-Z])\b",
        r"^\s*([A-Z])[\.\)]\s",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, head):
            choice = match.group(1)
            if choice in valid_choices:
                return choice
    return None


def _normalize_mcq_gold_answers(sample: BenchmarkSample) -> set[str]:
    valid_choices = {str(choice).upper() for choice in (sample.options or {})}
    normalized: set[str] = set()
    for answer in sample.answers:
        value = str(answer or "").strip()
        if not value:
            continue
        upper_value = value.upper()
        if upper_value in valid_choices:
            normalized.add(upper_value)
            continue
        for key, option_text in (sample.options or {}).items():
            if _normalize(value) == _normalize(option_text):
                normalized.add(str(key).upper())
    return normalized


def evaluate_prediction(sample: BenchmarkSample, prediction: str) -> dict[str, Any]:
    if sample.task == "mcq":
        valid_choices = set(sample.options or {})
        gold = _normalize_mcq_gold_answers(sample)
        predicted = _extract_mcq_choice(prediction, valid_choices)
        if not gold:
            return {
                "metric": "mcq_exact_choice",
                "evaluable": False,
                "predicted_choice": predicted,
                "gold_choices": [],
                "correct": None,
            }
        return {
            "metric": "mcq_exact_choice",
            "evaluable": True,
            "predicted_choice": predicted,
            "gold_choices": sorted(gold),
            "correct": predicted in gold if predicted is not None else False,
        }

    gold_answers = [str(answer).strip() for answer in sample.answers if str(answer or "").strip()]
    normalized_prediction = _normalize(prediction)
    if not gold_answers:
        return {
            "metric": "open_ended_em_f1",
            "prediction_for_eval": prediction,
            "exact_match": False,
            "em": 0.0,
            "f1": 0.0,
            "best_gold": None,
            "correct": False,
        }

    em_scores = [1.0 if normalized_prediction == _normalize(answer) else 0.0 for answer in gold_answers]
    f1_scores = [_token_f1(prediction, answer) for answer in gold_answers]
    best_idx = max(range(len(gold_answers)), key=lambda idx: (em_scores[idx], f1_scores[idx]))
    em = float(max(em_scores))
    f1 = float(max(f1_scores))
    return {
        "metric": "open_ended_em_f1",
        "prediction_for_eval": prediction,
        "exact_match": bool(em),
        "em": em,
        "f1": f1,
        "best_gold": gold_answers[best_idx],
        "correct": bool(em),
    }

#!/usr/bin/env python3
"""Cheap learnability screen for direct core-relation existence on gold endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "datasets/kg/rag2_statement_graph_relation_oracle_1k_v1"
DEFAULT_OUTPUT = ROOT / "datasets/kg/rag2_statement_graph_relation_learnability_tfidf_v1"
RUN_VERSION = "rag2_statement_graph_relation_learnability_tfidf_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mark_pair(claim: str, pair: Mapping[str, Any]) -> str:
    left, right = pair["left"], pair["right"]
    spans = [
        (int(left["start"]), int(left["end"]), "E1"),
        (int(right["start"]), int(right["end"]), "E2"),
    ]
    nonoverlap = spans[0][1] <= spans[1][0] or spans[1][1] <= spans[0][0]
    marked = claim
    if nonoverlap and all(0 <= start < end <= len(claim) for start, end, _ in spans):
        for start, end, label in sorted(spans, reverse=True):
            marked = marked[:start] + f" [{label}] " + marked[start:end] + f" [/{label}] " + marked[end:]
    return (
        marked
        + f" [LEFT] {left['surface']} [LEFT_TYPE] {left['kind']}"
        + f" [RIGHT] {right['surface']} [RIGHT_TYPE] {right['kind']}"
    )


def metrics(gold: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    tn = sum(g == 0 and p == 0 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pairs": len(gold), "positive": tp + fn, "negative": tn + fp,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2, "f1": f1,
    }


def main() -> None:
    args = parse_args()
    source = args.oracle_root / "oracle_pairs.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    split_counts = Counter(str(row["split"]) for row in rows)
    if split_counts != {"development": 512, "validation": 128, "test": 128}:
        raise RuntimeError(f"Unexpected frozen split: {dict(split_counts)}")

    examples: dict[str, list[dict[str, Any]]] = {"development": [], "validation": []}
    for row in rows:
        split = str(row["split"])
        if split not in examples:
            continue
        for pair in row["candidate_pairs"]:
            examples[split].append({
                "proposition_id": row["proposition_id"], "pair_id": pair["pair_id"],
                "source": row["source"], "text": mark_pair(str(row["claim"]), pair),
                "gold": int(bool(pair["gold_relations"])),
            })

    print(f"[stage 1/3] frozen input: development={len(examples['development']):,} validation={len(examples['validation']):,}; test untouched", flush=True)
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=60000)),
    ])
    train_text = [row["text"] for row in examples["development"]]
    train_gold = [row["gold"] for row in examples["development"]]
    valid_text = [row["text"] for row in examples["validation"]]
    valid_gold = [row["gold"] for row in examples["validation"]]
    train_x = features.fit_transform(train_text)
    valid_x = features.transform(valid_text)
    print(f"[stage 2/3] TF-IDF fitted: features={train_x.shape[1]:,}; fitting fixed class-balanced logistic regression", flush=True)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=17)
    model.fit(train_x, train_gold)
    probability = model.predict_proba(valid_x)[:, 1]
    prediction = [int(value >= 0.5) for value in probability]
    result = metrics(valid_gold, prediction)
    majority = metrics(valid_gold, [0] * len(valid_gold))
    passed_screen = result["balanced_accuracy"] >= 0.70 and result["recall"] >= 0.65 and result["specificity"] >= 0.70

    args.output_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = {
        "run_version": RUN_VERSION,
        "purpose": "screen whether proposition plus gold endpoints contains learnable signal before neural relation-gate training",
        "input_sha256": digest,
        "splits": {"development_pairs": len(train_gold), "validation_pairs": len(valid_gold), "test_status": "untouched"},
        "model": "word(1-2)+character(3-5) TF-IDF, fixed class-balanced logistic regression C=1.0, threshold=0.5",
        "majority_no_relation": majority,
        "validation": result,
        "screen_threshold": {"balanced_accuracy_min": 0.70, "recall_min": 0.65, "specificity_min": 0.70},
        "passed_screen": passed_screen,
        "next_action": "run bounded BiomedBERT tiny-overfit and held-out pilot" if passed_screen else "stop supervised relation-gate training; revise target/reference",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, score, pred in zip(examples["validation"], probability, prediction):
            handle.write(json.dumps({**row, "relation_probability": float(score), "prediction": pred}, ensure_ascii=False) + "\n")
    print(
        f"[stage 3/3] validation N={result['pairs']:,} BA={100*result['balanced_accuracy']:.2f}% "
        f"P={100*result['precision']:.2f}% R={100*result['recall']:.2f}% specificity={100*result['specificity']:.2f}% "
        f"decision={'PASS' if passed_screen else 'FAIL'} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()

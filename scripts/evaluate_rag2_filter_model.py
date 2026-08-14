from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from medrag.filtering.rag2_official import (
    LABEL_NAMES,
    convert_legacy_filter_input,
    resolve_label_token_ids,
)


DEFAULT_DATA_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "filtering"
    / "rag2"
    / "combined_medqa_medmcqa_fulltext"
    / "split_8_1_1"
    / "val.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RAG2 Flan-T5 filtering model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--scoring-method",
        choices=["generate", "log_likelihood", "special_token"],
        default="generate",
    )
    parser.add_argument("--input-format", choices=["auto", "legacy", "official"], default="auto")
    parser.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def load_rows(path: Path, max_samples: int | None, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0:
        return list(iter_rows(path))
    if sample_mode == "first":
        rows = []
        for row in iter_rows(path):
            rows.append(row)
            if len(rows) >= max_samples:
                break
        return rows

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_rows(path), start=1):
        if len(reservoir) < max_samples:
            reservoir.append(row)
            continue
        replace_idx = rng.randrange(idx)
        if replace_idx < max_samples:
            reservoir[replace_idx] = row
    rng.shuffle(reservoir)
    return reservoir


def normalize_label(text: Any) -> str:
    value = " ".join(str(text or "").lower().strip().split())
    value = re.sub(r"[^a-z ]+", "", value).strip()
    if "not helpful" in value or value in {"nothelpful", "unhelpful"}:
        return "not helpful"
    if "helpful" in value:
        return "helpful"
    return value


def batched(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def candidate_log_likelihood_scores(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    encoded_inputs: dict[str, torch.Tensor],
    candidates: list[str],
    normalization: str,
    device: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[int]]]:
    score_by_label: dict[str, list[float]] = {}
    sum_logprob_by_label: dict[str, list[float]] = {}
    token_count_by_label: dict[str, list[int]] = {}

    for candidate in candidates:
        target = tokenizer(
            text_target=[candidate] * int(encoded_inputs["input_ids"].shape[0]),
            padding=True,
            return_tensors="pt",
        ).to(device)
        labels = target["input_ids"]
        label_mask = labels.ne(tokenizer.pad_token_id)

        if hasattr(model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
        else:
            decoder_input_ids = model._shift_right(labels)  # type: ignore[attr-defined]

        outputs = model(
            input_ids=encoded_inputs["input_ids"],
            attention_mask=encoded_inputs.get("attention_mask"),
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        safe_labels = labels.masked_fill(~label_mask, 0)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * label_mask
        sum_log_probs = token_log_probs.sum(dim=-1)
        token_counts = label_mask.sum(dim=-1).clamp(min=1)
        if normalization == "mean":
            scores = sum_log_probs / token_counts
        else:
            scores = sum_log_probs

        score_by_label[candidate] = [float(x) for x in scores.detach().cpu()]
        sum_logprob_by_label[candidate] = [float(x) for x in sum_log_probs.detach().cpu()]
        token_count_by_label[candidate] = [int(x) for x in token_counts.detach().cpu()]

    return score_by_label, sum_logprob_by_label, token_count_by_label


def classification_metrics(
    label_counts: Counter[str],
    pred_counts: Counter[str],
    confusion: dict[str, Counter[str]],
    labels: list[str],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, float]] = {}
    recalls = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(label_counts[label]),
            "predicted": float(pred_counts[label]),
        }
        recalls.append(recall)
    return {
        "per_label": per_label,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data_file, args.max_samples, args.sample_mode, args.sample_seed)
    if not rows:
        raise ValueError(f"No rows loaded from {args.data_file}")

    dtype = torch.bfloat16 if args.bf16 and args.device.startswith("cuda") else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    label_token_ids = resolve_label_token_ids(tokenizer) if args.scoring_method == "special_token" else None
    use_official_input = args.input_format == "official" or (
        args.input_format == "auto" and args.scoring_method == "special_token"
    )

    total = 0
    correct = 0
    invalid = 0
    labels = ["helpful", "not helpful"]
    label_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output_handle = args.output_file.open("w", encoding="utf-8") if args.output_file else None
    try:
        with torch.inference_mode():
            for batch in tqdm(list(batched(rows, args.batch_size)), desc=args.scoring_method):
                inputs = [
                    convert_legacy_filter_input(row["input"]) if use_official_input else row["input"]
                    for row in batch
                ]
                targets = [normalize_label(row["target"]) for row in batch]
                encoded = tokenizer(
                    inputs,
                    truncation=True,
                    max_length=args.max_input_length,
                    padding=True,
                    return_tensors="pt",
                ).to(args.device)

                if args.scoring_method == "special_token":
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=1,
                        num_beams=1,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    if not generated.scores:
                        raise RuntimeError("RAG2 special-token generation returned no decoder scores.")
                    token_ids = label_token_ids or {}
                    score_tensor = generated.scores[0].float()[:, [
                        token_ids["helpful"],
                        token_ids["not helpful"],
                    ]]
                    prob_tensor = torch.softmax(score_tensor, dim=-1)
                    pred_indices = score_tensor.argmax(dim=-1)
                    predictions = [LABEL_NAMES[int(index)] for index in pred_indices]
                    raw_predictions = predictions
                    score_rows = []
                    for i in range(len(batch)):
                        score_rows.append(
                            {
                                "score_helpful": float(score_tensor[i, 0].detach().cpu()),
                                "score_not_helpful": float(score_tensor[i, 1].detach().cpu()),
                                "margin_helpful_minus_not_helpful": float(
                                    (score_tensor[i, 0] - score_tensor[i, 1]).detach().cpu()
                                ),
                                "prob_helpful_over_candidates": float(prob_tensor[i, 0].detach().cpu()),
                                "prob_not_helpful_over_candidates": float(prob_tensor[i, 1].detach().cpu()),
                            }
                        )
                elif args.scoring_method == "generate":
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                    )
                    raw_predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    predictions = [normalize_label(raw_pred) for raw_pred in raw_predictions]
                    score_rows = [{} for _ in predictions]
                else:
                    score_by_label, sum_logprob_by_label, token_count_by_label = candidate_log_likelihood_scores(
                        model=model,
                        tokenizer=tokenizer,
                        encoded_inputs=encoded,
                        candidates=labels,
                        normalization=args.score_normalization,
                        device=args.device,
                    )
                    score_tensor = torch.tensor([[score_by_label[label][i] for label in labels] for i in range(len(batch))])
                    prob_tensor = torch.softmax(score_tensor, dim=-1)
                    predictions = [labels[int(idx)] for idx in score_tensor.argmax(dim=-1)]
                    raw_predictions = predictions
                    score_rows = []
                    for i in range(len(batch)):
                        score_rows.append(
                            {
                                "score_helpful": score_by_label["helpful"][i],
                                "score_not_helpful": score_by_label["not helpful"][i],
                                "sum_logprob_helpful": sum_logprob_by_label["helpful"][i],
                                "sum_logprob_not_helpful": sum_logprob_by_label["not helpful"][i],
                                "token_count_helpful": token_count_by_label["helpful"][i],
                                "token_count_not_helpful": token_count_by_label["not helpful"][i],
                                "margin_helpful_minus_not_helpful": score_by_label["helpful"][i]
                                - score_by_label["not helpful"][i],
                                "prob_helpful_over_candidates": float(prob_tensor[i, 0]),
                                "prob_not_helpful_over_candidates": float(prob_tensor[i, 1]),
                            }
                        )

                for row, target, raw_pred, pred, scores in zip(batch, targets, raw_predictions, predictions, score_rows):
                    is_valid = pred in {"helpful", "not helpful"}
                    is_correct = pred == target
                    total += 1
                    correct += int(is_correct)
                    invalid += int(not is_valid)
                    label_counts[target] += 1
                    pred_counts[pred] += 1
                    confusion[target][pred] += 1
                    dataset = str(row.get("dataset") or "unknown")
                    by_dataset[dataset]["total"] += 1
                    by_dataset[dataset]["correct"] += int(is_correct)
                    if len(examples) < 30 or not is_correct:
                        item = {
                            "id": row.get("id"),
                            "dataset": dataset,
                            "source": row.get("source"),
                            "target": target,
                            "prediction": pred,
                            "raw_prediction": raw_pred,
                            "correct": is_correct,
                            **scores,
                        }
                        if len(examples) < 200:
                            examples.append(item)
                    if output_handle:
                        out = dict(row)
                        out["prediction"] = pred
                        out["raw_prediction"] = raw_pred
                        out["correct"] = is_correct
                        out.update(scores)
                        output_handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    finally:
        if output_handle:
            output_handle.close()

    summary = {
        "model_path": str(args.model_path),
        "data_file": str(args.data_file),
        "scoring_method": args.scoring_method,
        "input_format": "official" if use_official_input else "legacy",
        "score_normalization": args.score_normalization,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "rows": total,
        "accuracy": correct / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "label_counts": dict(label_counts),
        "prediction_counts": dict(pred_counts),
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
        **classification_metrics(label_counts, pred_counts, confusion, labels),
        "by_dataset": {
            dataset: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for dataset, counts in by_dataset.items()
        },
        "examples": examples[:30],
    }
    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import gc
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .core import CaseResult
from .progress import StageProgress


def _silence_noisy_loggers() -> None:
    for name in (
        "vllm",
        "vllm.engine",
        "vllm.executor",
        "vllm.worker",
        "vllm.config",
        "vllm.entrypoints",
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "tensorflow",
        "absl",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _cleanup_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _rows_from_results(results: list[CaseResult], exclude_datasets: set[str] | None = None) -> tuple[list[dict[str, Any]], list[int]]:
    exclude = {item.lower() for item in (exclude_datasets or set())}
    rows: list[dict[str, Any]] = []
    indices: list[int] = []
    for idx, result in enumerate(results):
        if result.sample.task != "open_ended":
            continue
        if result.sample.dataset.lower() in exclude:
            continue
        rows.append(
            {
                "dataset": result.sample.dataset,
                "id": result.sample.id,
                "question": result.sample.question,
                "answers": result.sample.answers,
                "pred_text": result.prediction,
            }
        )
        indices.append(idx)
    return rows, indices


class OpenEndedBERTScore:
    name = "bertscore"

    def __init__(
        self,
        model_type: str,
        lang: str | None = "en",
        batch_size: int = 32,
        num_layers: int | None = 24,
        rescale_with_baseline: bool = True,
        baseline_path: str | None = None,
        device: str | None = "cuda",
        use_fast_tokenizer: bool = False,
        show_progress: bool = True,
    ) -> None:
        self.model_type = model_type
        self.lang = lang
        self.batch_size = max(1, int(batch_size))
        self.num_layers = None if num_layers is None else int(num_layers)
        self.rescale_with_baseline = bool(rescale_with_baseline)
        self.baseline_path = baseline_path
        self.device = str(device).strip() if device not in (None, "", "null") else None
        self.use_fast_tokenizer = bool(use_fast_tokenizer)
        self.show_progress = bool(show_progress)

    def compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        from bert_score import score as bertscore_score  # type: ignore

        pair_preds: list[str] = []
        pair_refs: list[str] = []
        pair_owner: list[int] = []
        for idx, row in enumerate(rows):
            pred = _safe_str(row.get("pred_text")).strip()
            refs = [_safe_str(ref).strip() for ref in row.get("answers", []) if _safe_str(ref).strip()]
            if not pred or not refs:
                continue
            for ref in refs:
                pair_preds.append(pred)
                pair_refs.append(ref)
                pair_owner.append(idx)

        if not pair_preds:
            return {"total_scored": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "per_dataset": {}, "per_sample": [None] * len(rows)}

        kwargs: dict[str, Any] = {
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "rescale_with_baseline": self.rescale_with_baseline,
            "use_fast_tokenizer": self.use_fast_tokenizer,
            "verbose": self.show_progress,
        }
        if self.num_layers is not None:
            kwargs["num_layers"] = self.num_layers
        if self.lang:
            kwargs["lang"] = self.lang
        if self.device:
            kwargs["device"] = self.device
        if self.baseline_path:
            kwargs["baseline_path"] = self.baseline_path

        precision, recall, f1 = bertscore_score(pair_preds, pair_refs, **kwargs)
        p_vals = [float(value) for value in precision.tolist()]
        r_vals = [float(value) for value in recall.tolist()]
        f_vals = [float(value) for value in f1.tolist()]

        best_by_sample: dict[int, dict[str, float]] = {}
        for pair_idx, owner in enumerate(pair_owner):
            candidate = {"precision": p_vals[pair_idx], "recall": r_vals[pair_idx], "f1": f_vals[pair_idx]}
            current = best_by_sample.get(owner)
            if current is None or candidate["f1"] > current["f1"]:
                best_by_sample[owner] = candidate

        per_sample: list[dict[str, float] | None] = [None] * len(rows)
        for owner, values in best_by_sample.items():
            per_sample[owner] = dict(values)

        values = list(best_by_sample.values())
        by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
        for owner, values_item in best_by_sample.items():
            by_dataset[_safe_str(rows[owner].get("dataset", "unknown"))].append(values_item)

        per_dataset = {}
        for dataset, items in sorted(by_dataset.items()):
            total = len(items)
            per_dataset[dataset] = {
                "total_scored": total,
                "precision": sum(item["precision"] for item in items) / total,
                "recall": sum(item["recall"] for item in items) / total,
                "f1": sum(item["f1"] for item in items) / total,
            }

        total = len(values)
        return {
            "total_scored": total,
            "precision": sum(item["precision"] for item in values) / total,
            "recall": sum(item["recall"] for item in values) / total,
            "f1": sum(item["f1"] for item in values) / total,
            "per_dataset": per_dataset,
            "per_sample": per_sample,
        }


class OpenEndedBLEURT:
    name = "bleurt"

    def __init__(
        self,
        checkpoint: str,
        use_cpu: bool = True,
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.use_cpu = bool(use_cpu)
        self.batch_size = max(1, int(batch_size))
        self.show_progress = bool(show_progress)

    def compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.use_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        from bleurt import score as bleurt_score  # type: ignore

        scorer = bleurt_score.BleurtScorer(self.checkpoint)
        pair_preds: list[str] = []
        pair_refs: list[str] = []
        pair_owner: list[int] = []
        for idx, row in enumerate(rows):
            pred = _safe_str(row.get("pred_text")).strip()
            refs = [_safe_str(ref).strip() for ref in row.get("answers", []) if _safe_str(ref).strip()]
            if not pred or not refs:
                continue
            for ref in refs:
                pair_preds.append(pred)
                pair_refs.append(ref)
                pair_owner.append(idx)

        if not pair_preds:
            return {"total_scored": 0, "bleurt": 0.0, "per_dataset": {}, "per_sample": [None] * len(rows)}

        all_scores: list[float] = []
        progress = StageProgress(total=len(pair_preds), desc="BLEURT", enabled=self.show_progress)
        try:
            for start in range(0, len(pair_preds), self.batch_size):
                end = min(start + self.batch_size, len(pair_preds))
                scores = scorer.score(references=pair_refs[start:end], candidates=pair_preds[start:end])
                all_scores.extend(float(score) for score in scores)
                progress.update(end - start)
        finally:
            progress.close()

        best_by_sample: dict[int, float] = {}
        for pair_idx, owner in enumerate(pair_owner):
            score = all_scores[pair_idx]
            if owner not in best_by_sample or score > best_by_sample[owner]:
                best_by_sample[owner] = score

        per_sample: list[float | None] = [None] * len(rows)
        for owner, score in best_by_sample.items():
            per_sample[owner] = float(score)

        values = list(best_by_sample.values())
        by_dataset: dict[str, list[float]] = defaultdict(list)
        for owner, score in best_by_sample.items():
            by_dataset[_safe_str(rows[owner].get("dataset", "unknown"))].append(score)
        per_dataset = {
            dataset: {"total_scored": len(items), "bleurt": sum(items) / len(items)}
            for dataset, items in sorted(by_dataset.items())
            if items
        }
        total = len(values)
        return {
            "total_scored": total,
            "bleurt": sum(values) / total if total else 0.0,
            "per_dataset": per_dataset,
            "per_sample": per_sample,
        }


def _parse_score_1to5(text: str) -> float | None:
    if not text:
        return None
    word_scores = {
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
    }
    patterns = [
        r"\[\s*RESULT\s*\]\s*[:=]?\s*\**\s*([1-5](?:\.\d+)?)\b",
        r"\bSCORE\s*[:=]\s*\**\s*([1-5](?:\.\d+)?)\b",
        r'"score"\s*[:=]\s*"?([1-5](?:\.\d+)?)"?',
        r"\bscore\s*[:=]\s*([1-5](?:\.\d+)?)\b",
        r"\bscore\s+(?:is|would\s+be|should\s+be)\s*[:=]?\s*([1-5](?:\.\d+)?)\b",
        r"\brating\s*[:=]\s*([1-5](?:\.\d+)?)\b",
        r"\brate\s+(?:it|this|the\s+answer)?\s*(?:as|a)?\s*([1-5](?:\.\d+)?)\b",
        r"\b([1-5](?:\.\d+)?)\s*/\s*5\b",
        r"\b([1-5](?:\.\d+)?)\s+(?:out\s+of\s+5|points?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if 1.0 <= value <= 5.0:
                return value
    stripped = text.strip()
    match = re.match(r"^\s*\**\s*([1-5](?:\.\d+)?)\b", stripped)
    if match:
        value = float(match.group(1))
        if 1.0 <= value <= 5.0:
            return value
    word_match = re.search(r"\bSCORE\s*[:=]\s*\**\s*(one|two|three|four|five)\b", stripped, flags=re.IGNORECASE)
    if word_match:
        return word_scores[word_match.group(1).lower()]
    for line in stripped.splitlines()[:3]:
        line_match = re.fullmatch(r"\s*\**\s*([1-5](?:\.\d+)?)\s*\**\s*", line.strip())
        if line_match:
            value = float(line_match.group(1))
            if 1.0 <= value <= 5.0:
                return value
        word_line_match = re.fullmatch(r"\s*\**\s*(one|two|three|four|five)\s*\**\s*", line.strip(), flags=re.IGNORECASE)
        if word_line_match:
            return word_scores[word_line_match.group(1).lower()]
    return None


class VLLMJudgeEngine:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        use_chat_template: bool = True,
        system_prompt: str | None = None,
        use_tqdm: bool = False,
        use_flashinfer_sampler: bool = False,
    ) -> None:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1" if use_flashinfer_sampler else "0")
        _silence_noisy_loggers()
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.use_tqdm = bool(use_tqdm)
        self.use_chat_template = bool(use_chat_template)
        self.system_prompt = (system_prompt or "").strip()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop or ["<|eot_id|>"],
        )
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
            runner="generate",
        )
        self._tokenizer = None
        if self.use_chat_template:
            try:
                self._tokenizer = self.llm.get_tokenizer()
            except Exception:
                self._tokenizer = None

    def _format_prompt(self, prompt: str) -> tuple[str, str]:
        if not self.use_chat_template or self._tokenizer is None:
            return prompt, "raw"
        try:
            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(formatted), "chat_template"
        except Exception:
            return prompt, "raw_fallback"

    def predict_texts(self, prompts: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
        model_prompts: list[str] = []
        prompt_modes: list[str] = []
        for prompt in prompts:
            model_prompt, prompt_mode = self._format_prompt(prompt)
            model_prompts.append(model_prompt)
            prompt_modes.append(prompt_mode)
        outputs = self.llm.generate(model_prompts, self.sampling_params, use_tqdm=self.use_tqdm)
        texts: list[str] = []
        infos: list[dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            text = output.outputs[0].text.strip() if output.outputs else ""
            texts.append(text)
            infos.append(
                {
                    "model": self.model_path,
                    "prompt_mode": prompt_modes[idx] if idx < len(prompt_modes) else "unknown",
                }
            )
        return texts, infos

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        candidates = [
            (llm, "shutdown"),
            (llm, "close"),
            (getattr(llm, "llm_engine", None), "shutdown"),
            (getattr(llm, "llm_engine", None), "close"),
            (getattr(llm, "engine", None), "shutdown"),
            (getattr(llm, "engine", None), "close"),
        ]
        for obj, method in candidates:
            if obj is None:
                continue
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    continue
        self.llm = None
        gc.collect()


class OpenEndedPrometheus2Judge:
    name = "prometheus2"

    DEFAULT_RUBRIC = (
        "Judge the predicted answer as an absolute grade against the reference answer(s).\n"
        "Be conservative and reward only clear semantic agreement with the reference.\n"
        "Score 5: semantically equivalent, fully correct, and directly answers the question.\n"
        "Score 4: mostly correct, with only minor omission, wording difference, or harmless extra detail.\n"
        "Score 3: partially correct, but missing important facts or containing notable inaccuracies.\n"
        "Score 2: mostly incorrect, with only weak relation to the reference.\n"
        "Score 1: completely incorrect, irrelevant, or contradictory.\n"
        "Do not give a high score for topic similarity alone.\n"
    )

    def __init__(
        self,
        judge_engine: VLLMJudgeEngine,
        rubric: str | None = None,
        batch_size: int = 8,
        max_samples: int = 0,
        retry_unparsed: bool = True,
        retry_rounds: int = 3,
        parse_rate_threshold: float = 1.0,
        require_all_scored: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.judge_engine = judge_engine
        self.rubric = (rubric or self.DEFAULT_RUBRIC).strip()
        self.batch_size = max(1, int(batch_size))
        self.max_samples = int(max_samples)
        self.retry_unparsed = bool(retry_unparsed)
        self.retry_rounds = max(0, int(retry_rounds))
        self.parse_rate_threshold = float(parse_rate_threshold)
        self.require_all_scored = bool(require_all_scored)
        self.show_progress = bool(show_progress)

    def _build_prompt(self, question: str, prediction: str, refs: list[str]) -> str:
        refs_text = "\n".join(f"- {ref}" for ref in refs) if refs else "- (none)"
        return (
            "You are a fair and objective medical QA evaluator.\n"
            "Compare the predicted answer against the reference answer(s) for the given question.\n"
            "Assign one absolute score from 1 to 5 using the rubric.\n"
            "Be conservative: use 4 or 5 only when the prediction is clearly correct and directly answers the question.\n\n"
            "### Question\n"
            f"{question}\n\n"
            "### Reference Answer(s)\n"
            f"{refs_text}\n\n"
            "### Predicted Answer\n"
            f"{prediction}\n\n"
            "### Rubric\n"
            f"{self.rubric}\n\n"
            "### Required Output\n"
            "The first line must be exactly: SCORE: <1-5>\n"
            "The second line may be: REASON: <one short sentence>\n"
            "Do not output any other numbers before the SCORE line.\n"
        )

    def _build_score_only_prompt(self, question: str, prediction: str, refs: list[str], retry_round: int = 1) -> str:
        refs_text = "\n".join(f"- {ref}" for ref in refs) if refs else "- (none)"
        return (
            "Score the predicted answer against the reference answer(s) for the question.\n"
            "Return exactly one line in this format and nothing else: SCORE: <1-5>\n\n"
            f"Question: {question}\n\n"
            "Reference Answer(s):\n"
            f"{refs_text}\n\n"
            "Predicted Answer:\n"
            f"{prediction}\n\n"
            "Rubric:\n"
            f"{self.rubric}\n"
        )

    def compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        limit = self.max_samples if self.max_samples > 0 else len(rows)
        items: list[dict[str, Any]] = []
        for idx, row in enumerate(rows[:limit]):
            question = _safe_str(row.get("question")).strip()
            prediction = _safe_str(row.get("pred_text")).strip()
            refs = [_safe_str(ref).strip() for ref in row.get("answers", []) if _safe_str(ref).strip()]
            if question and prediction and refs:
                items.append({"row_index": idx, "dataset": _safe_str(row.get("dataset", "unknown")), "question": question, "prediction": prediction, "refs": refs})

        per_sample: list[dict[str, Any] | None] = [None] * len(rows)
        scored_rows: list[dict[str, Any]] = []
        retry_attempted = 0
        retry_scored = 0
        unparsed_items: list[dict[str, Any]] = []
        progress = StageProgress(total=len(items), desc="Prometheus2", enabled=self.show_progress)
        try:
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                prompts = [self._build_prompt(item["question"], item["prediction"], item["refs"]) for item in batch]
                judge_texts, judge_infos = self.judge_engine.predict_texts(prompts)

                retry_items: list[dict[str, Any]] = []
                for item, judge_text, judge_info in zip(batch, judge_texts, judge_infos):
                    score = _parse_score_1to5(judge_text)
                    if score is None:
                        retry_items.append({**item, "last_judge_text": judge_text, "last_judge_info": judge_info})
                        continue
                    scored_rows.append({"row_index": item["row_index"], "dataset": item["dataset"], "score_1to5": score})
                    per_sample[item["row_index"]] = {
                        "score_0to1": (score - 1.0) / 4.0,
                        "judge_text": judge_text,
                        "judge_info": judge_info,
                        "retried": False,
                    }

                unparsed_items.extend(retry_items)
                progress.update(len(batch))

            if self.retry_unparsed and unparsed_items:
                pending = unparsed_items
                for retry_round in range(1, self.retry_rounds + 1):
                    if not pending:
                        break
                    next_pending: list[dict[str, Any]] = []
                    for start in range(0, len(pending), self.batch_size):
                        batch = pending[start : start + self.batch_size]
                        retry_attempted += len(batch)
                        retry_prompts = [
                            self._build_score_only_prompt(item["question"], item["prediction"], item["refs"], retry_round=retry_round)
                            for item in batch
                        ]
                        retry_texts, retry_infos = self.judge_engine.predict_texts(retry_prompts)
                        for item, judge_text, judge_info in zip(batch, retry_texts, retry_infos):
                            score = _parse_score_1to5(judge_text)
                            if score is None:
                                next_pending.append({**item, "last_judge_text": judge_text, "last_judge_info": judge_info})
                                continue
                            retry_scored += 1
                            scored_rows.append({"row_index": item["row_index"], "dataset": item["dataset"], "score_1to5": score})
                            per_sample[item["row_index"]] = {
                                "score_0to1": (score - 1.0) / 4.0,
                                "judge_text": judge_text,
                                "judge_info": judge_info,
                                "retried": True,
                                "retry_round": retry_round,
                            }
                    pending = next_pending
        finally:
            progress.close()

        total_requested = len(items)
        total_scored = len(scored_rows)
        parse_rate = total_scored / total_requested if total_requested else 0.0
        if self.require_all_scored and total_scored != total_requested:
            missing = total_requested - total_scored
            missing_ids = [
                _safe_str(rows[item["row_index"]].get("id", item["row_index"]))
                for item in items
                if per_sample[item["row_index"]] is None
            ][:20]
            raise RuntimeError(
                f"Prometheus2 failed to parse {missing}/{total_requested} judge outputs; "
                f"missing sample ids: {missing_ids}. "
                "The evaluator already retried the missing rows with stricter score-only prompts. "
                "Increase --prometheus2-retry-rounds or --prometheus2-max-new-tokens, then rerun."
            )
        values = [row["score_1to5"] for row in scored_rows]
        mean_1to5 = sum(values) / len(values) if values else 0.0
        by_dataset: dict[str, list[float]] = defaultdict(list)
        requested_by_dataset: dict[str, int] = defaultdict(int)
        for item in items:
            requested_by_dataset[item["dataset"]] += 1
        for row in scored_rows:
            by_dataset[row["dataset"]].append(row["score_1to5"])

        per_dataset = {}
        for dataset in sorted(requested_by_dataset):
            dataset_values = by_dataset.get(dataset, [])
            dataset_mean = sum(dataset_values) / len(dataset_values) if dataset_values else 0.0
            requested = requested_by_dataset[dataset]
            per_dataset[dataset] = {
                "total_requested": requested,
                "total_scored": len(dataset_values),
                "parse_rate": len(dataset_values) / requested if requested else 0.0,
                "mean_score_0to1": (dataset_mean - 1.0) / 4.0,
            }

        return {
            "total_requested": total_requested,
            "total_scored": total_scored,
            "parse_rate": parse_rate,
            "parse_rate_threshold": self.parse_rate_threshold,
            "parse_rate_ok": parse_rate >= self.parse_rate_threshold if total_requested else True,
            "retry_attempted": retry_attempted,
            "retry_scored": retry_scored,
            "mean_score_0to1": (mean_1to5 - 1.0) / 4.0,
            "per_dataset": per_dataset,
            "per_sample": per_sample,
        }


def apply_open_ended_extra_metrics(results: list[CaseResult], config: dict[str, Any]) -> dict[str, Any]:
    exclude_datasets = {
        str(item).strip().lower()
        for item in config.get("open_ended_extra_metrics_exclude_datasets", ["pubmedqa"])
        if str(item).strip()
    }
    rows, result_indices = _rows_from_results(results, exclude_datasets=exclude_datasets)
    summary: dict[str, Any] = {}
    if exclude_datasets:
        summary["excluded_datasets"] = sorted(exclude_datasets)
    if not rows:
        return summary

    for result in results:
        if result.sample.task != "open_ended":
            continue
        result.evaluation.setdefault("bertscore_precision", None)
        result.evaluation.setdefault("bertscore_recall", None)
        result.evaluation.setdefault("bertscore_f1", None)
        result.evaluation.setdefault("bleurt", None)
        result.evaluation.setdefault("prometheus2_0to1", None)
        result.evaluation.setdefault("prometheus2_judge_text", None)

    show_progress = bool(config.get("show_progress", True))

    def _annotate_bertscore(metric_summary: dict[str, Any]) -> None:
        per_sample = metric_summary.get("per_sample")
        if not isinstance(per_sample, list):
            return
        for result_idx, sample_metric in zip(result_indices, per_sample):
            if not isinstance(sample_metric, dict):
                continue
            evaluation = results[result_idx].evaluation
            evaluation["bertscore_precision"] = _as_float(sample_metric.get("precision"))
            evaluation["bertscore_recall"] = _as_float(sample_metric.get("recall"))
            evaluation["bertscore_f1"] = _as_float(sample_metric.get("f1"))

    def _annotate_bleurt(metric_summary: dict[str, Any]) -> None:
        per_sample = metric_summary.get("per_sample")
        if not isinstance(per_sample, list):
            return
        for result_idx, sample_metric in zip(result_indices, per_sample):
            results[result_idx].evaluation["bleurt"] = _as_float(sample_metric)

    def _annotate_prometheus(metric_summary: dict[str, Any]) -> None:
        per_sample = metric_summary.get("per_sample")
        if not isinstance(per_sample, list):
            return
        for result_idx, sample_metric in zip(result_indices, per_sample):
            if not isinstance(sample_metric, dict):
                continue
            evaluation = results[result_idx].evaluation
            evaluation["prometheus2_0to1"] = _as_float(sample_metric.get("score_0to1"))
            evaluation["prometheus2_judge_text"] = sample_metric.get("judge_text")
            evaluation["prometheus2_retried"] = bool(sample_metric.get("retried"))
            if sample_metric.get("retry_round") is not None:
                evaluation["prometheus2_retry_round"] = sample_metric.get("retry_round")
            if sample_metric.get("judge_info") is not None:
                evaluation["prometheus2_judge_info"] = sample_metric.get("judge_info")

    if bool(config.get("bertscore_enabled", True)):
        start = time.time()
        logging.info("Extra evaluation start: BERTScore")
        bertscore_device = config.get("bertscore_device", "cuda")
        try:
            evaluator = OpenEndedBERTScore(
                model_type=str(config["bertscore_model_type"]),
                lang=config.get("bertscore_lang", "en"),
                batch_size=int(config.get("bertscore_batch_size", 32)),
                num_layers=config.get("bertscore_num_layers", 24),
                rescale_with_baseline=bool(config.get("bertscore_rescale_with_baseline", True)),
                baseline_path=config.get("bertscore_baseline_path"),
                device=bertscore_device,
                use_fast_tokenizer=bool(config.get("bertscore_use_fast_tokenizer", False)),
                show_progress=show_progress,
            )
            metric_summary = evaluator.compute(rows)
            summary["bertscore"] = metric_summary
            _annotate_bertscore(metric_summary)
        except Exception as exc:
            if str(bertscore_device).lower() not in ("cpu", "-1"):
                logging.warning("BERTScore on %s failed; retrying on CPU. Error: %s", bertscore_device, exc)
                _cleanup_gpu_memory()
                try:
                    evaluator = OpenEndedBERTScore(
                        model_type=str(config["bertscore_model_type"]),
                        lang=config.get("bertscore_lang", "en"),
                        batch_size=max(1, min(8, int(config.get("bertscore_batch_size", 32)))),
                        num_layers=config.get("bertscore_num_layers", 24),
                        rescale_with_baseline=bool(config.get("bertscore_rescale_with_baseline", True)),
                        baseline_path=config.get("bertscore_baseline_path"),
                        device="cpu",
                        use_fast_tokenizer=bool(config.get("bertscore_use_fast_tokenizer", False)),
                        show_progress=show_progress,
                    )
                    metric_summary = evaluator.compute(rows)
                    metric_summary["fallback_device"] = "cpu"
                    metric_summary["original_error"] = str(exc)
                    summary["bertscore"] = metric_summary
                    _annotate_bertscore(metric_summary)
                except Exception as cpu_exc:
                    summary["bertscore"] = {"error": str(cpu_exc), "original_error": str(exc)}
                    logging.exception("BERTScore CPU fallback failed")
            else:
                summary["bertscore"] = {"error": str(exc)}
                logging.exception("BERTScore evaluation failed")
        finally:
            _cleanup_gpu_memory()
        logging.info("Extra evaluation done: BERTScore (%.1fs)", time.time() - start)

    if bool(config.get("prometheus2_enabled", True)):
        start = time.time()
        logging.info("Extra evaluation start: Prometheus2")
        judge_engine = None
        try:
            judge_engine = VLLMJudgeEngine(
                model_path=str(config["prometheus2_model_path"]),
                max_model_len=int(config.get("prometheus2_max_model_len", 8192)),
                gpu_memory_utilization=float(config.get("prometheus2_gpu_memory_utilization", 0.90)),
                max_new_tokens=int(config.get("prometheus2_max_new_tokens", 96)),
                temperature=float(config.get("prometheus2_temperature", 0.0)),
                top_p=float(config.get("prometheus2_top_p", 1.0)),
                stop=list(config.get("prometheus2_stop", ["</s>", "<|im_end|>", "<|eot_id|>"])),
                use_chat_template=bool(config.get("prometheus2_use_chat_template", True)),
                system_prompt=config.get("prometheus2_system_prompt"),
                use_tqdm=bool(config.get("prometheus2_use_tqdm", False)),
                use_flashinfer_sampler=bool(config.get("prometheus2_use_flashinfer_sampler", False)),
            )
            evaluator = OpenEndedPrometheus2Judge(
                judge_engine=judge_engine,
                rubric=config.get("prometheus2_rubric"),
                batch_size=int(config.get("prometheus2_batch_size", 16)),
                max_samples=int(config.get("prometheus2_max_samples", 0)),
                retry_unparsed=bool(config.get("prometheus2_retry_unparsed", True)),
                retry_rounds=int(config.get("prometheus2_retry_rounds", 3)),
                parse_rate_threshold=float(config.get("prometheus2_parse_rate_threshold", 1.0)),
                require_all_scored=bool(config.get("prometheus2_require_all_scored", True)),
                show_progress=show_progress,
            )
            metric_summary = evaluator.compute(rows)
            summary["prometheus2"] = metric_summary
            _annotate_prometheus(metric_summary)
        except Exception as exc:
            summary["prometheus2"] = {"error": str(exc)}
            logging.exception("Prometheus2 evaluation failed")
            if bool(config.get("prometheus2_fail_on_error", True)):
                raise
        finally:
            if judge_engine is not None:
                judge_engine.close()
            _cleanup_gpu_memory()
        logging.info("Extra evaluation done: Prometheus2 (%.1fs)", time.time() - start)

    if bool(config.get("bleurt_enabled", True)):
        start = time.time()
        logging.info("Extra evaluation start: BLEURT")
        try:
            evaluator = OpenEndedBLEURT(
                checkpoint=str(config["bleurt_checkpoint"]),
                use_cpu=bool(config.get("bleurt_use_cpu", True)),
                batch_size=int(config.get("bleurt_batch_size", 128)),
                show_progress=show_progress,
            )
            metric_summary = evaluator.compute(rows)
            summary["bleurt"] = metric_summary
            _annotate_bleurt(metric_summary)
        except Exception as exc:
            summary["bleurt"] = {"error": str(exc)}
            logging.exception("BLEURT evaluation failed")
        finally:
            _cleanup_gpu_memory()
        logging.info("Extra evaluation done: BLEURT (%.1fs)", time.time() - start)

    return summary

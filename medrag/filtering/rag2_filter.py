from __future__ import annotations

import logging
import re
from copy import copy
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..core import BenchmarkSample, RetrievedDocument
from .rag2_official import (
    LABEL_NAMES,
    build_official_filter_input,
    format_options,
    resolve_label_token_ids,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _option_text(sample: BenchmarkSample) -> str:
    if not isinstance(sample.options, dict):
        return ""
    return "\n".join(f"{key}. {_clean_text(sample.options[key])}" for key in sorted(sample.options))


def _doc_text(doc: RetrievedDocument, max_chars: int) -> str:
    title = _clean_text(doc.title)
    text = _clean_text(doc.text)
    rendered = text or title
    if max_chars > 0 and len(rendered) > max_chars:
        return rendered[: max_chars - 3].rstrip() + "..."
    return rendered


def build_filter_input_from_evidence(
    sample: BenchmarkSample,
    evidence: Any,
    max_doc_chars: int,
    input_format: str = "legacy",
) -> str:
    """Build one filter prompt from an already selected evidence unit.

    Document filtering passes a full document here.  Attribution-window
    filtering passes a sentence-context window instead.  Both paths therefore
    retain exactly the same question/options and RAG² label-token contract.
    """

    rendered = _clean_text(evidence)
    if max_doc_chars > 0 and len(rendered) > max_doc_chars:
        rendered = rendered[: max_doc_chars - 3].rstrip() + "..."
    if input_format == "official":
        return build_official_filter_input(
            question=sample.question,
            options=format_options(sample.options),
            evidence=rendered,
        )
    if input_format != "legacy":
        raise ValueError(f"Unsupported RAG2 filter input format: {input_format}")
    return "\n\n".join(
        part
        for part in [
            "Decide whether the retrieved document is helpful for answering the medical multiple-choice question.",
            "Question:\n" + _clean_text(sample.question),
            "Options:\n" + _option_text(sample),
            "Retrieved document:\n" + rendered,
            "Answer with exactly one label: helpful or not helpful.",
        ]
        if part.strip()
    )


def build_filter_input(
    sample: BenchmarkSample,
    doc: RetrievedDocument,
    max_doc_chars: int,
    input_format: str = "legacy",
) -> str:
    return build_filter_input_from_evidence(
        sample,
        _doc_text(doc, max_doc_chars),
        max_doc_chars=0,
        input_format=input_format,
    )


def normalize_filter_label(text: Any) -> str:
    value = " ".join(str(text or "").lower().strip().split())
    value = value.replace("[", "").replace("]", "").replace("_", " ")
    value = re.sub(r"[^a-z ]+", "", value).strip()
    if "not helpful" in value or value in {"nothelpful", "unhelpful"}:
        return "not helpful"
    if "helpful" in value:
        return "helpful"
    return "not helpful"


class Rag2FlanT5Filter:
    def __init__(
        self,
        model_path: Path,
        batch_size: int = 128,
        max_input_length: int = 512,
        max_new_tokens: int = 8,
        max_doc_chars: int = 2600,
        device: str = "auto",
        bf16: bool = True,
        scoring_method: str = "generate",
        score_normalization: str = "mean",
        input_format: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.batch_size = max(1, int(batch_size))
        self.max_input_length = int(max_input_length)
        self.max_new_tokens = int(max_new_tokens)
        self.max_doc_chars = int(max_doc_chars)
        self.device = self._resolve_device(device)
        self.bf16 = bool(bf16)
        if scoring_method not in {"generate", "log_likelihood", "special_token"}:
            raise ValueError(f"Unsupported filter scoring_method: {scoring_method}")
        self.scoring_method = scoring_method
        if input_format == "auto":
            input_format = "official" if scoring_method == "special_token" else "legacy"
        if input_format not in {"legacy", "official"}:
            raise ValueError(f"Unsupported filter input_format: {input_format}")
        self.input_format = input_format
        if score_normalization not in {"mean", "sum"}:
            raise ValueError(f"Unsupported filter score_normalization: {score_normalization}")
        self.score_normalization = score_normalization

        logging.info("Loading RAG2 filter model: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if self.bf16 and self.device.type == "cuda" else None
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.label_token_ids = (
            resolve_label_token_ids(self.tokenizer) if self.scoring_method == "special_token" else None
        )
        logging.info("RAG2 filter model ready on %s", self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _candidate_scores(self, encoded_inputs: dict[str, torch.Tensor], candidates: list[str]) -> dict[str, torch.Tensor]:
        scores_by_label: dict[str, torch.Tensor] = {}
        batch_size = int(encoded_inputs["input_ids"].shape[0])
        for candidate in candidates:
            target = self.tokenizer(
                text_target=[candidate] * batch_size,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            labels = target["input_ids"]
            label_mask = labels.ne(self.tokenizer.pad_token_id)
            if hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            else:
                decoder_input_ids = self.model._shift_right(labels)  # type: ignore[attr-defined]

            outputs = self.model(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs.get("attention_mask"),
                decoder_input_ids=decoder_input_ids,
            )
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            safe_labels = labels.masked_fill(~label_mask, 0)
            token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            token_log_probs = token_log_probs * label_mask
            summed = token_log_probs.sum(dim=-1)
            if self.score_normalization == "mean":
                token_counts = label_mask.sum(dim=-1).clamp(min=1)
                scores_by_label[candidate] = summed / token_counts
            else:
                scores_by_label[candidate] = summed
        return scores_by_label

    def _score_inputs(self, inputs: list[str], progress_callback: Any | None = None) -> list[dict[str, float | str]]:
        labels = ["helpful", "not helpful"]
        scored: list[dict[str, float | str]] = []
        start = 0
        active_batch_size = self.batch_size
        while start < len(inputs):
            batch_inputs = inputs[start : start + active_batch_size]
            try:
                encoded = self.tokenizer(
                    batch_inputs,
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    if self.scoring_method == "special_token":
                        generated = self.model.generate(
                            **encoded,
                            max_new_tokens=1,
                            num_beams=1,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        if not generated.scores:
                            raise RuntimeError("RAG2 special-token generation returned no decoder scores.")
                        token_ids = self.label_token_ids or {}
                        score_tensor = generated.scores[0].float()[:, [
                            token_ids["helpful"],
                            token_ids["not helpful"],
                        ]]
                        probs = torch.softmax(score_tensor, dim=-1)
                        predictions = score_tensor.argmax(dim=-1)
                        margins = score_tensor[:, 0] - score_tensor[:, 1]
                        for idx in range(len(batch_inputs)):
                            pred_idx = int(predictions[idx].detach().cpu())
                            prediction = LABEL_NAMES[pred_idx]
                            scored.append(
                                {
                                    "prediction": prediction,
                                    "raw_prediction": prediction,
                                    "score_helpful": float(score_tensor[idx, 0].detach().cpu()),
                                    "score_not_helpful": float(score_tensor[idx, 1].detach().cpu()),
                                    "margin": float(margins[idx].detach().cpu()),
                                    "prob_helpful": float(probs[idx, 0].detach().cpu()),
                                }
                            )
                    elif self.scoring_method == "generate":
                        generated = self.model.generate(
                            **encoded,
                            max_new_tokens=self.max_new_tokens,
                            num_beams=1,
                            do_sample=False,
                        )
                        raw_predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                        predictions = [normalize_filter_label(raw_prediction) for raw_prediction in raw_predictions]
                        for prediction, raw_prediction in zip(predictions, raw_predictions):
                            is_helpful = prediction == "helpful"
                            scored.append(
                                {
                                    "prediction": prediction,
                                    "raw_prediction": raw_prediction,
                                    "score_helpful": 1.0 if is_helpful else 0.0,
                                    "score_not_helpful": 0.0 if is_helpful else 1.0,
                                    "margin": 1.0 if is_helpful else -1.0,
                                    "prob_helpful": 1.0 if is_helpful else 0.0,
                                }
                            )
                    else:
                        scores = self._candidate_scores(encoded, labels)
                        score_tensor = torch.stack([scores[label] for label in labels], dim=-1)
                        probs = torch.softmax(score_tensor, dim=-1)
                        predictions = score_tensor.argmax(dim=-1)
                        margins = score_tensor[:, 0] - score_tensor[:, 1]
                        for idx in range(len(batch_inputs)):
                            pred_idx = int(predictions[idx].detach().cpu())
                            scored.append(
                                {
                                    "prediction": labels[pred_idx],
                                    "raw_prediction": labels[pred_idx],
                                    "score_helpful": float(scores["helpful"][idx].detach().cpu()),
                                    "score_not_helpful": float(scores["not helpful"][idx].detach().cpu()),
                                    "margin": float(margins[idx].detach().cpu()),
                                    "prob_helpful": float(probs[idx, 0].detach().cpu()),
                                }
                            )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or active_batch_size <= 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                active_batch_size = max(1, active_batch_size // 2)
                logging.warning("RAG2 filter OOM; retrying with filter_batch_size=%s", active_batch_size)
                continue
            if progress_callback is not None:
                progress_callback(len(batch_inputs))
            start += len(batch_inputs)
        return scored

    def score_evidences(
        self,
        samples: list[BenchmarkSample],
        evidences: list[str],
        progress_callback: Any | None = None,
    ) -> list[dict[str, float | str]]:
        """Score arbitrary evidence units while reusing the loaded filter.

        This is intentionally public so the MCQ evaluator can score sentence
        windows in bounded question batches without reloading Flan-T5 for
        every batch.  The existing document-filter path still uses
        :meth:`filter_batch` unchanged.
        """

        if len(samples) != len(evidences):
            raise ValueError(f"Sample/evidence length mismatch: {len(samples)} != {len(evidences)}")
        inputs = [
            build_filter_input_from_evidence(
                sample,
                evidence,
                self.max_doc_chars,
                input_format=self.input_format,
            )
            for sample, evidence in zip(samples, evidences)
        ]
        return self._score_inputs(inputs, progress_callback=progress_callback)

    def score_filter_inputs(
        self,
        inputs: list[str],
        progress_callback: Any | None = None,
    ) -> list[dict[str, float | str]]:
        """Score pre-rendered filter prompts without rebuilding their question block."""

        return self._score_inputs(inputs, progress_callback=progress_callback)

    def filter_batch(
        self,
        samples: list[BenchmarkSample],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        fill_to_top_k: bool = False,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        inputs: list[str] = []
        spans: list[tuple[int, int]] = []
        for sample, docs in zip(samples, candidate_lists):
            start = len(inputs)
            inputs.extend(
                build_filter_input(sample, doc, self.max_doc_chars, input_format=self.input_format)
                for doc in docs
            )
            spans.append((start, len(inputs)))

        flat_scores = self._score_inputs(inputs, progress_callback=progress_callback) if inputs else []
        outputs: list[list[RetrievedDocument]] = []
        for docs, (start, end) in zip(candidate_lists, spans):
            rescored: list[RetrievedDocument] = []
            for doc, scores in zip(docs, flat_scores[start:end]):
                # Keep every filtering decision on the reranked candidate list for analysis.
                doc.filter_prediction = str(scores["prediction"])
                doc.filter_score = float(scores["margin"])
                doc.filter_prob_helpful = float(scores["prob_helpful"])
                updated = copy(doc)
                rescored.append(updated)
            selected = [doc for doc in rescored if doc.filter_prediction == "helpful"][:top_k]
            if fill_to_top_k and len(selected) < top_k:
                selected_ids = {id(doc) for doc in selected}
                for doc in rescored:
                    if id(doc) in selected_ids:
                        continue
                    selected.append(doc)
                    selected_ids.add(id(doc))
                    if len(selected) >= top_k:
                        break
            for rank, doc in enumerate(selected, start=1):
                doc.filter_rank = rank
            outputs.append(selected)
        return outputs

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class DatasetRoutedRag2Filter:
    """Route MCQ datasets to separate Flan-T5 filters while preserving sample order."""

    def __init__(
        self,
        model_paths: dict[str, Path],
        dataset_routes: dict[str, str],
        **filter_kwargs: Any,
    ) -> None:
        self.model_paths = {str(key): Path(value) for key, value in model_paths.items()}
        self.dataset_routes = {str(key): str(value) for key, value in dataset_routes.items()}
        self.filter_kwargs = dict(filter_kwargs)

    def filter_batch(
        self,
        samples: list[BenchmarkSample],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        fill_to_top_k: bool = False,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        if len(samples) != len(candidate_lists):
            raise ValueError(f"Sample/candidate length mismatch: {len(samples)} != {len(candidate_lists)}")
        outputs: list[list[RetrievedDocument] | None] = [None] * len(samples)
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            route = self.dataset_routes.get(sample.dataset)
            if route is None:
                raise ValueError(f"No RAG2 filter route configured for dataset: {sample.dataset}")
            if route not in self.model_paths:
                raise ValueError(f"No RAG2 filter model configured for route: {route}")
            groups.setdefault(route, []).append(index)

        for route, indices in groups.items():
            logging.info(
                "Filtering route=%s datasets=%s samples=%s model=%s",
                route,
                sorted({samples[index].dataset for index in indices}),
                len(indices),
                self.model_paths[route],
            )
            filterer = Rag2FlanT5Filter(model_path=self.model_paths[route], **self.filter_kwargs)
            try:
                routed_outputs = filterer.filter_batch(
                    samples=[samples[index] for index in indices],
                    candidate_lists=[candidate_lists[index] for index in indices],
                    top_k=top_k,
                    fill_to_top_k=fill_to_top_k,
                    progress_callback=progress_callback,
                )
            finally:
                filterer.close()
            for index, docs in zip(indices, routed_outputs):
                outputs[index] = docs
        return [docs or [] for docs in outputs]

    def close(self) -> None:
        return


class LazyFilter:
    def __init__(self, factory: Any, name: str = "filter") -> None:
        self.factory = factory
        self.name = name
        self._impl: Any | None = None

    @property
    def impl(self) -> Any:
        if self._impl is None:
            logging.info("Lazy-loading %s.", self.name)
            self._impl = self.factory()
        return self._impl

    def filter_batch(
        self,
        samples: list[BenchmarkSample],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        fill_to_top_k: bool = False,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        return self.impl.filter_batch(
            samples=samples,
            candidate_lists=candidate_lists,
            top_k=top_k,
            fill_to_top_k=fill_to_top_k,
            progress_callback=progress_callback,
        )

    def close(self) -> None:
        if self._impl is None:
            return
        close_fn = getattr(self._impl, "close", None)
        if callable(close_fn):
            close_fn()
        self._impl = None


class NoOpFilter:
    def filter_batch(
        self,
        samples: list[BenchmarkSample],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        fill_to_top_k: bool = False,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        del samples
        if progress_callback is not None:
            progress_callback(sum(len(docs) for docs in candidate_lists))
        outputs = []
        for docs in candidate_lists:
            selected = [copy(doc) for doc in docs[:top_k]]
            for rank, doc in enumerate(selected, start=1):
                doc.filter_rank = rank
            outputs.append(selected)
        return outputs

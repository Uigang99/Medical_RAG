from __future__ import annotations

import logging
from copy import copy
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core import RetrievedDocument


class MedCPTCrossEncoderReranker:
    def __init__(
        self,
        model_path: Path,
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "auto",
        attn_implementation: str = "eager",
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._resolve_device(device)
        self.attn_implementation = attn_implementation
        logging.info("Loading MedCPT cross-encoder: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        kwargs: dict[str, Any] = {"local_files_only": True, "attn_implementation": attn_implementation}
        if self.device.type == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        except Exception:
            if attn_implementation == "eager":
                raise
            logging.warning("Retrying MedCPT cross-encoder with eager attention.")
            kwargs["attn_implementation"] = "eager"
            self.attn_implementation = "eager"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        logging.info("MedCPT cross-encoder ready on %s", self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _article_text(doc: RetrievedDocument) -> str:
        if doc.title and doc.text and not str(doc.text).startswith(str(doc.title)):
            return f"{doc.title}. {doc.text}"
        return doc.text or doc.title or ""

    def _score_pairs(self, pairs: list[list[str]], progress_callback: Any | None = None) -> list[float]:
        scores: list[float] = []
        start = 0
        active_batch_size = self.batch_size
        while start < len(pairs):
            batch_pairs = pairs[start : start + active_batch_size]
            try:
                encoded = self.tokenizer(
                    batch_pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
                with torch.inference_mode():
                    try:
                        logits = self.model(**encoded).logits.reshape(-1)
                    except RuntimeError as exc:
                        msg = str(exc)
                        is_cudnn_plan_error = "cudnn_frontend" in msg or "No valid execution plans" in msg
                        if self.attn_implementation == "eager" or not is_cudnn_plan_error:
                            raise
                        logging.warning("Retrying MedCPT cross-encoder batch with eager attention after cuDNN plan error.")
                        self._reload_with_eager_attention()
                        logits = self.model(**encoded).logits.reshape(-1)
                scores.extend(logits.float().cpu().tolist())
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or active_batch_size <= 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                active_batch_size = max(1, active_batch_size // 2)
                logging.warning("MedCPT cross-encoder OOM; retrying with rerank_batch_size=%s", active_batch_size)
                continue
            if progress_callback is not None:
                progress_callback(len(batch_pairs))
            start += len(batch_pairs)
        return scores

    def _reload_with_eager_attention(self) -> None:
        self.close()
        self.attn_implementation = "eager"
        kwargs: dict[str, Any] = {"local_files_only": True, "attn_implementation": "eager"}
        if self.device.type == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, docs: list[RetrievedDocument]) -> list[float]:
        pairs = [[query, self._article_text(doc)] for doc in docs]
        return self._score_pairs(pairs)

    def rerank(self, query: str, docs: list[RetrievedDocument], top_k: int) -> list[RetrievedDocument]:
        if not docs:
            return []
        scores = self.score(query, docs)
        rescored: list[RetrievedDocument] = []
        for doc, score in zip(docs, scores):
            updated = copy(doc)
            updated.rerank_score = float(score)
            rescored.append(updated)
        rescored.sort(key=lambda doc: doc.rerank_score if doc.rerank_score is not None else float("-inf"), reverse=True)
        selected = rescored[:top_k]
        for rank, doc in enumerate(selected, start=1):
            doc.rerank_rank = rank
        return selected

    def rerank_batch(
        self,
        queries: list[str],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        pairs: list[list[str]] = []
        spans: list[tuple[int, int]] = []
        for query, docs in zip(queries, candidate_lists):
            start = len(pairs)
            pairs.extend([query, self._article_text(doc)] for doc in docs)
            spans.append((start, len(pairs)))

        flat_scores = self._score_pairs(pairs, progress_callback=progress_callback) if pairs else []
        outputs: list[list[RetrievedDocument]] = []
        for docs, (start, end) in zip(candidate_lists, spans):
            rescored: list[RetrievedDocument] = []
            for doc, score in zip(docs, flat_scores[start:end]):
                updated = copy(doc)
                updated.rerank_score = float(score)
                rescored.append(updated)
            rescored.sort(
                key=lambda doc: doc.rerank_score if doc.rerank_score is not None else float("-inf"),
                reverse=True,
            )
            selected = rescored[:top_k]
            for rank, doc in enumerate(selected, start=1):
                doc.rerank_rank = rank
            outputs.append(selected)
        return outputs

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class LazyReranker:
    def __init__(self, factory: Any, name: str = "reranker") -> None:
        self.factory = factory
        self.name = name
        self._impl: Any | None = None

    @property
    def impl(self) -> Any:
        if self._impl is None:
            logging.info("Lazy-loading %s.", self.name)
            self._impl = self.factory()
        return self._impl

    def rerank_batch(
        self,
        queries: list[str],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        return self.impl.rerank_batch(
            queries,
            candidate_lists,
            top_k=top_k,
            progress_callback=progress_callback,
        )

    def close(self) -> None:
        if self._impl is None:
            return
        close_fn = getattr(self._impl, "close", None)
        if callable(close_fn):
            close_fn()
        self._impl = None


class NoOpReranker:
    def rerank_batch(
        self,
        queries: list[str],
        candidate_lists: list[list[RetrievedDocument]],
        top_k: int,
        progress_callback: Any | None = None,
    ) -> list[list[RetrievedDocument]]:
        del queries
        if progress_callback is not None:
            progress_callback(sum(len(docs) for docs in candidate_lists))
        return [docs[:top_k] for docs in candidate_lists]

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .core import BenchmarkSample, CaseResult, RetrievedDocument
from .evaluation import evaluate_prediction
from .filtering.rag2_filter import LazyFilter, NoOpFilter, Rag2FlanT5Filter
from .generation.base import TextGenerator
from .postprocessing import postprocess_generation_text
from .progress import StageProgress
from .prompts import build_prompt_request
from .retrieval.faiss_retriever import FaissMedCPTRetriever
from .reranking.medcpt_cross_encoder import MedCPTCrossEncoderReranker, NoOpReranker


NO_RAG = "no_rag"
BASELINE_RAG = "baseline_rag"
RERANK_RAG = "rerank_rag"
FILTER_RAG = "filter_rag"
DEFAULT_CASES = [NO_RAG, BASELINE_RAG, RERANK_RAG]
AVAILABLE_CASES = [NO_RAG, BASELINE_RAG, RERANK_RAG, FILTER_RAG]


@dataclass(frozen=True)
class RunnerConfig:
    cases: list[str]
    top_k: int = 5
    top_n: int = 50
    max_doc_chars: int = 1200
    retrieval_batch_size: int = 32
    retrieval_progress_chunk_size: int = 16
    generation_batch_size: int = 8
    show_progress: bool = True
    retrieval_cache_dir: Path | None = None
    rerank_cache_dir: Path | None = None
    filter_cache_dir: Path | None = None
    use_retrieval_cache: bool = True
    use_rerank_cache: bool = True
    use_filter_cache: bool = True
    filter_candidate_scope: str = "top_n"
    filter_fill_to_top_k: bool = False


def _sample_cache_key(sample: BenchmarkSample) -> str:
    return f"{sample.task}::{sample.collection}::{sample.dataset}::{sample.split}::{sample.id}::{sample.row_idx}"


def _doc_from_dict(row: dict[str, Any]) -> RetrievedDocument:
    return RetrievedDocument(
        source=str(row.get("source") or ""),
        local_id=int(row.get("local_id", -1)),
        db_id=str(row.get("db_id") or ""),
        corpus_id=row.get("corpus_id"),
        chunk_id=row.get("chunk_id"),
        doc_id=row.get("doc_id"),
        title=row.get("title"),
        text=str(row.get("text") or ""),
        retrieval_score=float(row.get("retrieval_score", 0.0)),
        retrieval_rank=row.get("retrieval_rank"),
        rerank_score=row.get("rerank_score"),
        rerank_rank=row.get("rerank_rank"),
        filter_score=row.get("filter_score"),
        filter_rank=row.get("filter_rank"),
        filter_prediction=row.get("filter_prediction"),
        filter_prob_helpful=row.get("filter_prob_helpful"),
        metadata=row.get("metadata") or {},
    )


class _DocListCache:
    def __init__(self, cache_dir: Path | None, filename: str, enabled: bool = True) -> None:
        self.enabled = bool(enabled and cache_dir is not None)
        self.path = cache_dir / filename if cache_dir is not None else None
        self.rows: dict[str, list[RetrievedDocument]] = {}
        if self.enabled and self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        self.rows[str(row["key"])] = [_doc_from_dict(item) for item in row.get("docs") or []]
                    except Exception:
                        continue
            logging.debug("Loaded %s cached rows from %s", len(self.rows), self.path)

    def get(self, key: str, min_docs: int) -> list[RetrievedDocument] | None:
        if not self.enabled:
            return None
        docs = self.rows.get(key)
        if docs is None or len(docs) < min_docs:
            return None
        return docs

    def put(self, key: str, docs: list[RetrievedDocument]) -> None:
        if not self.enabled or self.path is None:
            return
        if key in self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows[key] = docs
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"key": key, "docs": [doc.to_dict(include_text=True) for doc in docs]},
                    ensure_ascii=False,
                )
                + "\n"
            )


class BasicRagRunner:
    def __init__(
        self,
        config: RunnerConfig,
        generator: TextGenerator,
        retriever: FaissMedCPTRetriever | None = None,
        reranker: MedCPTCrossEncoderReranker | NoOpReranker | None = None,
        filterer: Rag2FlanT5Filter | LazyFilter | NoOpFilter | None = None,
    ) -> None:
        self.config = config
        self.generator = generator
        self.retriever = retriever
        self.reranker = reranker or NoOpReranker()
        self.filterer = filterer or NoOpFilter()

    @staticmethod
    def _sample_query_text(sample: BenchmarkSample) -> str:
        lines = [sample.question]
        if sample.task == "mcq":
            lines.extend(sample.option_lines())
        return "\n".join(line for line in lines if line)

    def _retrieve_if_needed(
        self,
        query_vectors: np.ndarray,
        samples: list[BenchmarkSample],
    ) -> list[list[RetrievedDocument]]:
        if (
            BASELINE_RAG not in self.config.cases
            and RERANK_RAG not in self.config.cases
            and FILTER_RAG not in self.config.cases
        ):
            return [[] for _ in samples]
        if self.retriever is None:
            raise RuntimeError("Retriever is required for RAG cases.")
        retrieve_k = self.config.top_n if RERANK_RAG in self.config.cases or FILTER_RAG in self.config.cases else self.config.top_k
        cache = _DocListCache(
            self.config.retrieval_cache_dir,
            "retrieval.jsonl",
            enabled=self.config.use_retrieval_cache,
        )
        all_docs: list[list[RetrievedDocument] | None] = [None for _ in samples]
        missing_indices: list[int] = []
        for idx, sample in enumerate(samples):
            cached_docs = cache.get(_sample_cache_key(sample), min_docs=retrieve_k)
            if cached_docs is not None:
                all_docs[idx] = cached_docs[:retrieve_k]
            else:
                missing_indices.append(idx)

        progress = StageProgress(total=len(samples), desc="Retrieval", enabled=self.config.show_progress)
        try:
            cached_count = len(samples) - len(missing_indices)
            if cached_count:
                progress.update(cached_count)
            for offset in range(0, len(missing_indices), self.config.retrieval_batch_size):
                batch_indices = missing_indices[offset : offset + self.config.retrieval_batch_size]
                logging.debug("Retrieving batch %s/%s (top_k=%s)", offset, len(missing_indices), retrieve_k)
                batch_vectors = query_vectors[batch_indices]
                completed_in_batch = 0

                def on_retrieved(n: int) -> None:
                    nonlocal completed_in_batch
                    delta = max(0, int(n))
                    if not delta:
                        return
                    completed_in_batch += delta
                    progress.update(delta)

                batch_docs = self.retriever.retrieve_batch(
                    batch_vectors,
                    top_k=retrieve_k,
                    progress_callback=on_retrieved,
                    progress_chunk_size=self.config.retrieval_progress_chunk_size,
                )
                for idx, docs in zip(batch_indices, batch_docs):
                    all_docs[idx] = docs
                    cache.put(_sample_cache_key(samples[idx]), docs)
                if completed_in_batch < len(batch_indices):
                    progress.update(len(batch_indices) - completed_in_batch)
        finally:
            progress.close()
        return [docs or [] for docs in all_docs]

    def _rerank_if_needed(
        self,
        samples: list[BenchmarkSample],
        retrieved_docs: list[list[RetrievedDocument]],
    ) -> list[list[RetrievedDocument]]:
        if RERANK_RAG not in self.config.cases and FILTER_RAG not in self.config.cases:
            return [[] for _ in samples]
        rerank_top_k = self.config.top_n if FILTER_RAG in self.config.cases else self.config.top_k
        cache = _DocListCache(
            self.config.rerank_cache_dir,
            "rerank.jsonl",
            enabled=self.config.use_rerank_cache,
        )
        reranked: list[list[RetrievedDocument] | None] = [None for _ in samples]
        missing_indices: list[int] = []
        cached_count = 0
        for idx, sample in enumerate(samples):
            cached_docs = cache.get(_sample_cache_key(sample), min_docs=rerank_top_k)
            if cached_docs is not None:
                reranked[idx] = cached_docs[:rerank_top_k]
                cached_count += 1
            else:
                missing_indices.append(idx)
        progress = StageProgress(total=len(samples), desc="Reranking", enabled=self.config.show_progress)
        if cached_count:
            progress.update(cached_count)
        if not missing_indices:
            progress.close()
            return [docs or [] for docs in reranked]

        queries = [self._sample_query_text(samples[idx]) for idx in missing_indices]
        candidates = [retrieved_docs[idx] for idx in missing_indices]
        total_pairs = sum(len(docs) for docs in candidates)
        logging.info("Reranking %s queries / %s query-document pairs", len(queries), total_pairs)

        pair_progress = 0
        query_progress = 0

        def on_pairs_done(n: int) -> None:
            nonlocal pair_progress, query_progress
            if progress:
                pair_progress += max(0, int(n))
                target_queries = int(pair_progress / max(1, total_pairs) * len(missing_indices))
                delta = max(0, target_queries - query_progress)
                if delta:
                    progress.update(delta)
                    query_progress += delta

        try:
            reranked_missing = self.reranker.rerank_batch(
                queries,
                candidates,
                top_k=rerank_top_k,
                progress_callback=on_pairs_done,
            )
            if query_progress < len(missing_indices):
                progress.update(len(missing_indices) - query_progress)
        finally:
            progress.close()
        for idx, docs in zip(missing_indices, reranked_missing):
            reranked[idx] = docs
            cache.put(_sample_cache_key(samples[idx]), docs)
        return [docs or [] for docs in reranked]

    def _filter_if_needed(
        self,
        samples: list[BenchmarkSample],
        reranked_docs: list[list[RetrievedDocument]],
    ) -> list[list[RetrievedDocument]]:
        if FILTER_RAG not in self.config.cases:
            return [[] for _ in samples]
        cache = _DocListCache(
            self.config.filter_cache_dir,
            "filter.jsonl",
            enabled=self.config.use_filter_cache,
        )
        filtered: list[list[RetrievedDocument] | None] = [None for _ in samples]
        missing_indices: list[int] = []
        cached_count = 0
        cache_min_docs = self.config.top_k if self.config.filter_fill_to_top_k else 0
        for idx, sample in enumerate(samples):
            cached_docs = cache.get(_sample_cache_key(sample), min_docs=cache_min_docs)
            if cached_docs is not None:
                filtered[idx] = cached_docs[: self.config.top_k]
                cached_count += 1
            else:
                missing_indices.append(idx)

        progress = StageProgress(total=len(samples), desc="Filtering", enabled=self.config.show_progress)
        if cached_count:
            progress.update(cached_count)
        if not missing_indices:
            progress.close()
            return [docs or [] for docs in filtered]

        batch_samples = [samples[idx] for idx in missing_indices]
        if self.config.filter_candidate_scope == "top_k":
            candidates = [reranked_docs[idx][: self.config.top_k] for idx in missing_indices]
        elif self.config.filter_candidate_scope == "top_n":
            candidates = [reranked_docs[idx][: self.config.top_n] for idx in missing_indices]
        else:
            raise ValueError(f"Unsupported filter_candidate_scope: {self.config.filter_candidate_scope}")
        total_pairs = sum(len(docs) for docs in candidates)
        logging.info("Filtering %s queries / %s query-document pairs", len(batch_samples), total_pairs)

        pair_progress = 0
        query_progress = 0

        def on_pairs_done(n: int) -> None:
            nonlocal pair_progress, query_progress
            pair_progress += max(0, int(n))
            target_queries = int(pair_progress / max(1, total_pairs) * len(missing_indices))
            delta = max(0, target_queries - query_progress)
            if delta:
                progress.update(delta)
                query_progress += delta

        try:
            filtered_missing = self.filterer.filter_batch(
                samples=batch_samples,
                candidate_lists=candidates,
                top_k=self.config.top_k,
                fill_to_top_k=self.config.filter_fill_to_top_k,
                progress_callback=on_pairs_done,
            )
            if query_progress < len(missing_indices):
                progress.update(len(missing_indices) - query_progress)
        finally:
            progress.close()
        for idx, docs in zip(missing_indices, filtered_missing):
            filtered[idx] = docs
            cache.put(_sample_cache_key(samples[idx]), docs)
        return [docs or [] for docs in filtered]

    def _generate_in_batches(self, prompts, case_id: str):
        batch_size = self.config.generation_batch_size
        if batch_size <= 0:
            batch_size = len(prompts)
        outputs = []
        iterator = range(0, len(prompts), batch_size)
        progress = StageProgress(total=len(prompts), desc="Generation", enabled=self.config.show_progress)
        try:
            for start in iterator:
                end = min(start + batch_size, len(prompts))
                logging.debug("Generating %s batch %s:%s", case_id, start, end)
                batch_outputs = self.generator.generate_batch(prompts[start:end])
                outputs.extend(batch_outputs)
                progress.update(len(batch_outputs))
        finally:
            progress.close()
        return outputs

    def _close_retriever(self) -> None:
        if self.retriever is None:
            return
        close_fn = getattr(self.retriever, "close", None)
        if callable(close_fn):
            logging.debug("Closing retriever before reranking/generation.")
            close_fn()
        self.retriever = None

    def run(
        self,
        samples: list[BenchmarkSample],
        query_vectors: np.ndarray,
    ) -> list[CaseResult]:
        if len(samples) != len(query_vectors):
            raise ValueError(f"Sample/query vector length mismatch: {len(samples)} != {len(query_vectors)}")

        try:
            retrieved_docs = self._retrieve_if_needed(query_vectors, samples)
            self._close_retriever()
            reranked_docs = self._rerank_if_needed(samples, retrieved_docs)
            if RERANK_RAG in self.config.cases or FILTER_RAG in self.config.cases:
                close_fn = getattr(self.reranker, "close", None)
                if callable(close_fn):
                    logging.debug("Closing reranker before generation.")
                    close_fn()
            filtered_docs = self._filter_if_needed(samples, reranked_docs)
            if FILTER_RAG in self.config.cases:
                close_fn = getattr(self.filterer, "close", None)
                if callable(close_fn):
                    logging.debug("Closing filter before generation.")
                    close_fn()
            results: list[CaseResult] = []

            for case_id in self.config.cases:
                logging.info("Running generation case: %s", case_id)
                prompts = []
                initial_by_sample: list[list[RetrievedDocument]] = []
                reranked_by_sample: list[list[RetrievedDocument]] = []
                final_by_sample: list[list[RetrievedDocument]] = []

                for idx, sample in enumerate(samples):
                    if case_id == NO_RAG:
                        initial_docs: list[RetrievedDocument] = []
                        reranked_candidates: list[RetrievedDocument] = []
                        final_docs: list[RetrievedDocument] = []
                    elif case_id == BASELINE_RAG:
                        initial_docs = retrieved_docs[idx][: self.config.top_k]
                        reranked_candidates = []
                        final_docs = initial_docs
                    elif case_id == RERANK_RAG:
                        initial_docs = retrieved_docs[idx][: self.config.top_n]
                        reranked_candidates = reranked_docs[idx][: self.config.top_n]
                        final_docs = reranked_docs[idx][: self.config.top_k]
                    elif case_id == FILTER_RAG:
                        initial_docs = retrieved_docs[idx][: self.config.top_n]
                        reranked_candidates = reranked_docs[idx][: self.config.top_n]
                        final_docs = filtered_docs[idx][: self.config.top_k]
                    else:
                        raise ValueError(f"Unsupported case: {case_id}")

                    prompts.append(
                        build_prompt_request(
                            sample=sample,
                            case_id=case_id,
                            docs=final_docs,
                            max_doc_chars=self.config.max_doc_chars,
                        )
                    )
                    initial_by_sample.append(initial_docs)
                    reranked_by_sample.append(reranked_candidates)
                    final_by_sample.append(final_docs)

                generations = self._generate_in_batches(prompts, case_id=case_id)
                eval_progress = StageProgress(total=len(samples), desc="Evaluation", enabled=self.config.show_progress)
                try:
                    for sample, generation, initial_docs, reranked_candidates, final_docs in zip(
                        samples, generations, initial_by_sample, reranked_by_sample, final_by_sample
                    ):
                        raw_prediction = generation.raw_text if generation.raw_text is not None else generation.text
                        prediction = postprocess_generation_text(raw_prediction)
                        results.append(
                            CaseResult(
                                case_id=case_id,
                                sample=sample,
                                prediction=prediction,
                                prompt=generation.prompt,
                                initial_documents=initial_docs,
                                final_documents=final_docs,
                                evaluation=evaluate_prediction(sample, prediction),
                                raw_prediction=raw_prediction,
                                reranked_documents=reranked_candidates,
                            )
                        )
                        eval_progress.update(1)
                finally:
                    eval_progress.close()
            return results
        finally:
            pass


def results_to_jsonable(results: list[CaseResult], include_doc_text: bool = True) -> list[dict[str, Any]]:
    return [result.to_dict(include_doc_text=include_doc_text) for result in results]

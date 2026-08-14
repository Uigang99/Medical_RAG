from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSample:
    row_idx: int
    id: str
    task: str
    collection: str
    dataset: str
    split: str
    question: str
    options: dict[str, str] | None
    answer: str | None
    answers: list[str]
    raw: dict[str, Any] = field(repr=False)

    def option_lines(self) -> list[str]:
        if not self.options:
            return []
        return [f"{key}. {self.options[key]}" for key in sorted(self.options)]


@dataclass
class RetrievedDocument:
    source: str
    local_id: int
    db_id: str
    corpus_id: str | None
    chunk_id: str | None
    doc_id: str | None
    title: str | None
    text: str
    retrieval_score: float
    retrieval_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    filter_score: float | None = None
    filter_rank: int | None = None
    filter_prediction: str | None = None
    filter_prob_helpful: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def stable_id(self) -> str:
        return self.corpus_id or self.chunk_id or self.db_id or f"{self.source}:{self.local_id}"

    def text_for_context(self, max_chars: int) -> str:
        pieces = []
        if self.title:
            pieces.append(str(self.title).strip())
        if self.text:
            pieces.append(str(self.text).strip())
        text = "\n".join(piece for piece in pieces if piece)
        if max_chars > 0 and len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        item = {
            "source": self.source,
            "local_id": self.local_id,
            "db_id": self.db_id,
            "corpus_id": self.corpus_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "filter_score": self.filter_score,
            "filter_rank": self.filter_rank,
            "filter_prediction": self.filter_prediction,
            "filter_prob_helpful": self.filter_prob_helpful,
        }
        if include_text:
            item["text"] = self.text
        generation_context = self.metadata.get("generation_context") if isinstance(self.metadata, dict) else None
        if isinstance(generation_context, dict):
            item["generation_context"] = generation_context
        return item


@dataclass(frozen=True)
class PromptRequest:
    sample_id: str
    case_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rendered(self) -> str:
        return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in self.messages)


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt: str
    raw_text: str | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None


@dataclass
class CaseResult:
    case_id: str
    sample: BenchmarkSample
    prediction: str
    prompt: str
    initial_documents: list[RetrievedDocument]
    final_documents: list[RetrievedDocument]
    evaluation: dict[str, Any]
    raw_prediction: str | None = None
    reranked_documents: list[RetrievedDocument] = field(default_factory=list)

    def to_dict(self, include_doc_text: bool = True) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "sample": {
                "row_idx": self.sample.row_idx,
                "id": self.sample.id,
                "task": self.sample.task,
                "collection": self.sample.collection,
                "dataset": self.sample.dataset,
                "split": self.sample.split,
                "question": self.sample.question,
                "options": self.sample.options,
                "answer": self.sample.answer,
                "answers": self.sample.answers,
            },
            "prediction": self.prediction,
            "raw_prediction": self.raw_prediction,
            "evaluation": self.evaluation,
            "prompt": self.prompt,
            "initial_retrieved_doc_ids": [doc.stable_id for doc in self.initial_documents],
            "reranked_doc_ids": [doc.stable_id for doc in self.reranked_documents],
            "final_context_doc_ids": [doc.stable_id for doc in self.final_documents],
            "initial_documents": [doc.to_dict(include_text=include_doc_text) for doc in self.initial_documents],
            "reranked_documents": [doc.to_dict(include_text=include_doc_text) for doc in self.reranked_documents],
            "final_documents": [doc.to_dict(include_text=include_doc_text) for doc in self.final_documents],
        }

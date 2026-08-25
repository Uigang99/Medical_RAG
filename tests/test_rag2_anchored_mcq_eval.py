from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medrag.core import BenchmarkSample, GenerationOutput, RetrievedDocument
from medrag.rag2_anchored_trace import (
    FORMAT_INSTRUCTION,
    PROMPT_VERSION,
    RATIONALE_HEADER,
)
from scripts.run_rag2_mcq_eval import (
    active_document_prompt_version,
    active_no_rag_prompt_version,
    build_document_messages_for_request,
    generate_rag_answers,
)


ROW = {
    "question": "Which vitamin is supplied primarily by animal sources?",
    "options": {
        "A": "Vitamin B12",
        "B": "Vitamin C",
        "C": "Vitamin B7",
        "D": "Vitamin D",
    },
    "answer": "A",
}


def sample() -> BenchmarkSample:
    return BenchmarkSample(
        row_idx=0,
        id="sample-0",
        task="mcq",
        collection="unified",
        dataset="medqa",
        split="test",
        question=ROW["question"],
        options=ROW["options"],
        answer="A",
        answers=["A"],
        raw=ROW,
    )


def args() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_profile="paper_compatible_three_anchor",
        answer_decision_mode="free_generation",
    )


class AnchoredMcqEvalTests(unittest.TestCase):
    def test_anchored_eval_uses_the_frozen_prompt_version(self) -> None:
        namespace = args()
        self.assertEqual(active_no_rag_prompt_version(namespace), PROMPT_VERSION)
        self.assertEqual(active_document_prompt_version(namespace), PROMPT_VERSION)

    def test_anchored_eval_renders_body_only_documents_without_numbering(self) -> None:
        messages = build_document_messages_for_request(
            args(),
            sample(),
            [
                {"source": "pubmed", "title": "Metadata title", "text": "Evidence one."},
                {"source": "pmc", "title": "Another title", "text": "Evidence two."},
            ],
            max_doc_chars=0,
            format_retry=False,
            selected_answer=None,
            choice_only=False,
        )
        content = messages[0]["content"]
        self.assertIn(FORMAT_INSTRUCTION, content)
        self.assertIn("Documents:\nEvidence one.\n\nEvidence two.", content)
        self.assertNotIn("Metadata title", content)
        self.assertNotIn("pubmed", content)
        self.assertNotIn("[1]", content)
        self.assertIn(RATIONALE_HEADER.strip(), content)

    def test_anchored_eval_applies_per_document_character_limit(self) -> None:
        messages = build_document_messages_for_request(
            args(),
            sample(),
            [{"text": "abcdefghij"}, {"text": "klmnopqrst"}],
            max_doc_chars=8,
            format_retry=False,
            selected_answer=None,
            choice_only=False,
        )
        self.assertIn("abcde...\n\nklmno...", messages[0]["content"])

    def test_anchored_eval_generates_rationale_then_one_constrained_choice(self) -> None:
        class FakeGenerator:
            def __init__(self) -> None:
                self.prefixes: list[str] = []

            def generate_batch(self, requests):
                return [
                    GenerationOutput(
                        text="Animal foods are the principal natural source of vitamin B12.",
                        prompt=request.rendered + "\n" + RATIONALE_HEADER,
                        raw_text="Animal foods are the principal natural source of vitamin B12.",
                    )
                    for request in requests
                ]

            def generate_allowed_single_token_continuations(self, prefixes):
                self.prefixes = list(prefixes)
                return [GenerationOutput(text="A", prompt=prefix, raw_text="A") for prefix in prefixes]

            def close(self) -> None:
                return

        namespace = SimpleNamespace(
            prompt_profile="paper_compatible_three_anchor",
            answer_decision_mode="free_generation",
            generation_batch_size=8,
            max_doc_chars=0,
            document_packing="fixed_chars",
            dense_query_mode="rationale",
            case="filter_rag",
            format_retry_attempts=0,
        )
        document = RetrievedDocument(
            source="pubmed",
            local_id=1,
            db_id="pubmed:1",
            corpus_id="pubmed:1",
            chunk_id=None,
            doc_id=None,
            title="Ignored title",
            text="Vitamin B12 is obtained predominantly from animal-derived foods.",
            retrieval_score=1.0,
            rerank_score=1.0,
            rerank_rank=1,
            filter_prediction="helpful",
        )
        fake = FakeGenerator()
        with patch("scripts.run_rag2_mcq_eval.build_generator", return_value=fake):
            results, details = generate_rag_answers(
                namespace,
                [sample()],
                [[document]],
                [[document]],
                [[document]],
                ["cached no-RAG rationale plus answer"],
                {},
            )
        self.assertEqual(results[0].prediction, "A")
        self.assertTrue(results[0].evaluation["correct"])
        self.assertIn("### END OF REASONING ###\nFinal answer: (A) Vitamin B12", results[0].raw_prediction)
        self.assertEqual(details["medqa::test::sample-0::0"]["final_answer"], "A")
        self.assertTrue(fake.prefixes[0].endswith("### END OF REASONING ###\nFinal answer: ("))

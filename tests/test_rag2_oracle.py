from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from medrag.core import BenchmarkSample, GenerationOutput, RetrievedDocument
from medrag.rag2_oracle import (
    canonicalize_rag2_labels,
    deterministic_question_sample,
    hidden_policy_name,
    oracle_document_is_helpful,
)
from scripts.evaluate_rag2_oracle_label_topk_sweep import (
    finalize_generations,
    policies,
    prompt_request,
    prompt_versions,
    repair_terminal_generations,
)
from scripts.run_rag2_mcq_eval import apply_oracle_labels, sample_key


def sample() -> BenchmarkSample:
    raw = {
        "question": "Which vitamin is supplied primarily by animal sources?",
        "options": {
            "A": "Vitamin B12",
            "B": "Vitamin C",
            "C": "Vitamin B7",
            "D": "Vitamin D",
        },
        "answer": "A",
    }
    return BenchmarkSample(
        row_idx=0,
        id="medqa:test:0",
        task="mcq",
        collection="medqa",
        dataset="medqa",
        split="test",
        question=raw["question"],
        options=raw["options"],
        answer="A",
        answers=["A"],
        raw=raw,
    )


class FakeTerminalGenerator:
    def __init__(self, choice: str = "B") -> None:
        self.choice = choice
        self.prefixes: list[str] = []

    def generate_allowed_single_token_continuations(
        self, prefixes: list[str]
    ) -> list[GenerationOutput]:
        self.prefixes = prefixes
        return [GenerationOutput(text=self.choice, prompt=prefix) for prefix in prefixes]


class Rag2OracleTests(unittest.TestCase):
    def test_direct_choice_uses_hidden_extraction_prompt_and_exact_choice_grammar(self) -> None:
        args = Namespace(answer_decision_mode="constrained_choice")
        request = prompt_request(args, sample(), [], case_id="no_rag", max_doc_chars=0)
        self.assertIn("Do not provide an explanation", request.messages[0]["content"])
        self.assertIn("Context:\nNone", request.messages[0]["content"])
        self.assertEqual(request.metadata["structured_regex"], r" (A|B|C|D)")
        self.assertEqual(
            prompt_versions(args),
            {
                "no_rag": "rag2_fixed_direct_choice_context_v1",
                "documents": "rag2_fixed_direct_choice_context_v1",
            },
        )

    def test_direct_choice_finalization_requires_one_valid_option(self) -> None:
        args = Namespace(answer_decision_mode="constrained_choice")
        output = GenerationOutput(text=" A", prompt="PROMPT")
        finalized = finalize_generations(args, FakeTerminalGenerator(), [sample()], [output])
        self.assertEqual(finalized[0][1:], ("A", "constrained_choice"))
        with self.assertRaisesRegex(RuntimeError, "direct-choice generation failed"):
            finalize_generations(
                args,
                FakeTerminalGenerator(),
                [sample()],
                [GenerationOutput(text="!", prompt="PROMPT")],
            )

    def test_policy_selection_can_run_hidden_threshold_without_rag2(self) -> None:
        args = Namespace(include_rag2=False, hidden_thresholds=[0.4])
        self.assertEqual(policies(args), [("hidden_tau_0p4", 0.4)])

    def test_terminal_repair_canonicalizes_answer_already_in_response(self) -> None:
        generator = FakeTerminalGenerator()
        generation = GenerationOutput(
            text="B12 is primarily found in animal products. The final answer is A.",
            prompt="PROMPT",
        )
        repaired = repair_terminal_generations(generator, [sample()], [generation])
        output, prediction, source = repaired[0]
        self.assertEqual(prediction, "A")
        self.assertEqual(source, "canonicalized_primary_answer")
        self.assertTrue(output.text.endswith("Therefore, the answer is (A) Vitamin B12."))
        self.assertEqual(generator.prefixes, [])

    def test_terminal_repair_uses_one_token_choice_only_when_answer_is_absent(self) -> None:
        generator = FakeTerminalGenerator(choice="B")
        generation = GenerationOutput(
            text="Vitamin B12 is found in animal products, while vitamin C is abundant in fruit.",
            prompt="PROMPT",
        )
        repaired = repair_terminal_generations(generator, [sample()], [generation])
        output, prediction, source = repaired[0]
        self.assertEqual(prediction, "B")
        self.assertEqual(source, "constrained_one_token_fallback")
        self.assertTrue(output.text.endswith("Therefore, the answer is (B) Vitamin C."))
        self.assertEqual(
            generator.prefixes,
            [f"PROMPT{generation.text}\nTherefore, the answer is ("],
        )

    def test_deterministic_sample_is_input_order_independent(self) -> None:
        values = ["q4", "q1", "q3", "q2"]
        first = deterministic_question_sample(values, dataset="medqa", limit=2, seed=42)
        second = deterministic_question_sample(reversed(values), dataset="medqa", limit=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_rag2_duplicate_prefers_only_quality_passing_replacement(self) -> None:
        rows = [
            {
                "sample_id": "q1",
                "doc_rank": 1,
                "doc_stable_id": "d1",
                "pseudo_label": "Excluded",
                "quality_pass": False,
            },
            {
                "sample_id": "q1",
                "doc_rank": 1,
                "doc_stable_id": "d1",
                "pseudo_label": "Helpful",
                "quality_pass": True,
            },
        ]
        labels, audit = canonicalize_rag2_labels(
            rows, selected_sample_ids={"q1"}, max_rank=8
        )
        self.assertEqual(labels["q1"]["d1"], "Helpful")
        self.assertEqual(audit["duplicate_keys"], 1)
        self.assertEqual(audit["valid_replacements"], 1)

    def test_multiple_valid_duplicate_is_rejected(self) -> None:
        rows = [
            {
                "sample_id": "q1",
                "doc_rank": 1,
                "doc_stable_id": "d1",
                "pseudo_label": label,
                "quality_pass": True,
            }
            for label in ("Helpful", "Not Helpful")
        ]
        with self.assertRaisesRegex(ValueError, "Multiple quality-passing"):
            canonicalize_rag2_labels(rows, selected_sample_ids={"q1"}, max_rank=8)

    def test_oracle_policy_boundaries(self) -> None:
        self.assertTrue(
            oracle_document_is_helpful(
                policy="rag2", rag2_label="Helpful", hidden_projection=None, hidden_threshold=None
            )
        )
        self.assertFalse(
            oracle_document_is_helpful(
                policy="rag2", rag2_label="Discard", hidden_projection=None, hidden_threshold=None
            )
        )
        self.assertTrue(
            oracle_document_is_helpful(
                policy="hidden_tau_0p2",
                rag2_label=None,
                hidden_projection=0.2001,
                hidden_threshold=0.2,
            )
        )
        self.assertFalse(
            oracle_document_is_helpful(
                policy="hidden_tau_0p2",
                rag2_label=None,
                hidden_projection=0.2,
                hidden_threshold=0.2,
            )
        )
        self.assertEqual(hidden_policy_name(0.2), "hidden_tau_0p2")

    def test_dynamic_oracle_join_uses_document_identity_when_rank_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.jsonl"
            labels_path.write_text(
                json.dumps(
                    {
                        "sample_key": sample_key(sample()),
                        "doc_rank": 37,
                        "doc_stable_id": "doc-1",
                        "pseudo_label": "Helpful",
                        "quality_pass": True,
                        "delta_ppl": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            document = RetrievedDocument(
                source="pubmed",
                local_id=1,
                db_id="doc-1",
                corpus_id=None,
                chunk_id=None,
                doc_id=None,
                title=None,
                text="Evidence",
                retrieval_score=1.0,
                rerank_score=2.0,
                rerank_rank=1,
            )
            args = Namespace(oracle_labels_path=labels_path, oracle_policy="rag2")
            result = apply_oracle_labels(args, [sample()], [[document]])
            self.assertEqual(result[0][0].filter_prediction, "helpful")
            self.assertEqual(result[0][0].filter_score, 0.5)

    def test_semantic_oracle_direct_and_supporting_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "semantic_labels.jsonl"
            documents = []
            label_rows = []
            for rank, (stable_id, semantic_label) in enumerate(
                (("direct", "direct_support"), ("supporting", "supporting_evidence"), ("noise", "no_evidence")),
                start=1,
            ):
                document = RetrievedDocument(
                    source="pubmed",
                    local_id=rank,
                    db_id=stable_id,
                    corpus_id=None,
                    chunk_id=None,
                    doc_id=None,
                    title=None,
                    text="Evidence",
                    retrieval_score=1.0,
                    rerank_score=2.0,
                    rerank_rank=rank,
                )
                documents.append(document)
                label_rows.append(
                    {
                        "sample_key": sample_key(sample()),
                        "doc_rank": rank,
                        "doc_stable_id": document.stable_id,
                        "semantic_label": semantic_label,
                        "confidence": 0.9,
                        "topic_relation": "related",
                    }
                )
            labels_path.write_text(
                "".join(json.dumps(row) + "\n" for row in label_rows),
                encoding="utf-8",
            )

            direct_args = Namespace(oracle_labels_path=labels_path, oracle_policy="semantic_direct")
            direct_result = apply_oracle_labels(direct_args, [sample()], [[*documents]])[0]
            self.assertEqual(
                [document.filter_prediction for document in direct_result],
                ["helpful", "not helpful", "not helpful"],
            )
            self.assertIsNone(direct_result[0].filter_score)
            self.assertEqual(direct_result[0].metadata["oracle_filter"]["semantic_confidence"], 0.9)

            broad_args = Namespace(
                oracle_labels_path=labels_path,
                oracle_policy="semantic_direct_supporting",
            )
            broad_result = apply_oracle_labels(broad_args, [sample()], [[*documents]])[0]
            self.assertEqual(
                [document.filter_prediction for document in broad_result],
                ["helpful", "helpful", "not helpful"],
            )

    def test_semantic_oracle_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "semantic_labels.jsonl"
            document = RetrievedDocument(
                source="pubmed",
                local_id=1,
                db_id="doc-1",
                corpus_id=None,
                chunk_id=None,
                doc_id=None,
                title=None,
                text="Evidence",
                retrieval_score=1.0,
                rerank_score=2.0,
                rerank_rank=1,
            )
            labels_path.write_text(
                json.dumps(
                    {
                        "sample_key": sample_key(sample()),
                        "doc_rank": 1,
                        "doc_stable_id": document.stable_id,
                        "semantic_label": "unknown",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(oracle_labels_path=labels_path, oracle_policy="semantic_direct")
            with self.assertRaisesRegex(ValueError, "Invalid semantic oracle label"):
                apply_oracle_labels(args, [sample()], [[document]])

    def test_behavioral_subset_oracle_uses_explicit_selected_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "subset_labels.jsonl"
            documents = [
                RetrievedDocument(
                    source="pubmed",
                    local_id=rank,
                    db_id=stable_id,
                    corpus_id=None,
                    chunk_id=None,
                    doc_id=None,
                    title=None,
                    text="Evidence",
                    retrieval_score=1.0,
                    rerank_score=2.0,
                    rerank_rank=rank,
                )
                for rank, stable_id in enumerate(("selected", "rejected"), start=1)
            ]
            labels_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "sample_key": sample_key(sample()),
                            "doc_rank": rank,
                            "doc_stable_id": document.stable_id,
                            "selection_policy": "behavioral_best_semantic_candidates",
                            "selected": rank == 1,
                            "pseudo_label": "Helpful" if rank == 1 else "Not Helpful",
                            "selected_subset_gold_margin": 1.25,
                            "selected_subset_size": 1,
                            "candidate_semantic_labels": [
                                "direct_support",
                                "supporting_evidence",
                            ],
                        }
                    )
                    + "\n"
                    for rank, document in enumerate(documents, start=1)
                ),
                encoding="utf-8",
            )
            args = Namespace(
                oracle_labels_path=labels_path,
                oracle_policy="behavioral_best_semantic_candidates",
            )
            result = apply_oracle_labels(args, [sample()], [[*documents]])[0]
            self.assertEqual(
                [document.filter_prediction for document in result],
                ["helpful", "not helpful"],
            )
            self.assertIsNone(result[0].filter_score)
            self.assertEqual(
                result[0].metadata["oracle_filter"]["behavioral_subset_gold_margin"],
                1.25,
            )

    def test_behavioral_subset_oracle_rejects_policy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "subset_labels.jsonl"
            document = RetrievedDocument(
                source="pubmed",
                local_id=1,
                db_id="doc-1",
                corpus_id=None,
                chunk_id=None,
                doc_id=None,
                title=None,
                text="Evidence",
                retrieval_score=1.0,
                rerank_score=2.0,
                rerank_rank=1,
            )
            labels_path.write_text(
                json.dumps(
                    {
                        "sample_key": sample_key(sample()),
                        "doc_rank": 1,
                        "doc_stable_id": document.stable_id,
                        "selection_policy": "behavioral_best_direct",
                        "selected": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(
                oracle_labels_path=labels_path,
                oracle_policy="behavioral_best_semantic_candidates",
            )
            with self.assertRaisesRegex(ValueError, "policy mismatch"):
                apply_oracle_labels(args, [sample()], [[document]])


if __name__ == "__main__":
    unittest.main()

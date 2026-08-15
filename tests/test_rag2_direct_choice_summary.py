from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medrag.core import BenchmarkSample
from scripts.run_rag2_mcq_eval import prompt_request, uses_free_terminal_generation
from scripts.summarize_rag2_direct_choice_all_mcq import DATASETS, summarize_rows


class DirectChoiceSummaryTests(unittest.TestCase):
    def test_direct_choice_bypasses_terminal_grammar_and_repair(self) -> None:
        direct = SimpleNamespace(
            answer_decision_mode="constrained_choice",
            prompt_profile="paper_exact_terminal",
        )
        rationale = SimpleNamespace(
            answer_decision_mode="free_generation",
            prompt_profile="paper_exact_terminal",
        )
        self.assertFalse(uses_free_terminal_generation(direct))
        self.assertTrue(uses_free_terminal_generation(rationale))

    def test_direct_choice_request_uses_exact_hidden_label_prompt_without_regex(self) -> None:
        raw = {
            "question": "Which vitamin is mainly obtained from animal products?",
            "options": {"A": "B12", "B": "C", "C": "B7", "D": "K"},
            "answer": "A",
        }
        sample = BenchmarkSample(
            row_idx=0,
            id="medqa:0",
            task="mcq",
            collection="unified",
            dataset="medqa",
            split="test",
            question=raw["question"],
            options=raw["options"],
            answer="A",
            answers=["A"],
            raw=raw,
        )
        args = SimpleNamespace(
            answer_decision_mode="constrained_choice",
            prompt_profile="paper_exact_terminal",
            document_packing="fixed_chars",
        )
        request = prompt_request(args, sample, [], "no_rag", 0)
        self.assertNotIn("structured_regex", request.metadata)
        self.assertIn("Do not provide an explanation", request.messages[0]["content"])
        self.assertIn("Context:\nNone", request.messages[0]["content"])

    def test_summary_uses_pooled_mmlu_and_eight_dataset_macro(self) -> None:
        rows = []
        for index, dataset in enumerate(DATASETS):
            rows.append(
                {
                    "sample": {"id": f"{dataset}:0", "dataset": dataset},
                    "evaluation": {"correct": index % 2 == 0},
                    "context_document_count": index,
                }
            )
        summary = summarize_rows(rows)
        self.assertEqual(summary["questions"], 8)
        self.assertEqual(summary["correct"], 4)
        self.assertAlmostEqual(summary["micro_accuracy"], 0.5)
        self.assertAlmostEqual(summary["macro_accuracy"], 0.5)
        self.assertAlmostEqual(summary["mmlu_pooled_accuracy"], 0.5)
        self.assertAlmostEqual(summary["mean_context_documents"], 3.5)


if __name__ == "__main__":
    unittest.main()

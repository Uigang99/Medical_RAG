from __future__ import annotations

import unittest

from medrag.core import BenchmarkSample, GenerationOutput
from medrag.rag2_oracle import (
    canonicalize_rag2_labels,
    deterministic_question_sample,
    hidden_policy_name,
    oracle_document_is_helpful,
)
from scripts.evaluate_rag2_oracle_label_topk_sweep import repair_terminal_generations


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


if __name__ == "__main__":
    unittest.main()

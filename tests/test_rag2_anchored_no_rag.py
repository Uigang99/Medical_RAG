from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_no_rag_train import artifact_row  # noqa: E402
from precompute_rag2_rationale_embeddings import (  # noqa: E402
    resolve_rationale_query,
    retrieval_query_contract,
)
from extract_rag2_anchored_no_rag_features import (  # noqa: E402
    RUN_VERSION as FEATURE_RUN_VERSION,
    complete_valid,
)


class AnchoredNoRAGTests(unittest.TestCase):
    def test_artifact_preserves_complete_retrieval_query(self) -> None:
        response = (
            "Rationale:\nClinical reasoning.\n### END OF REASONING ###\n"
            "Final answer: (B) Beta"
        )
        trace = {
            "sample_id": "medqa:train:000001",
            "dataset": "medqa",
            "question": "Question?",
            "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
            "gold_answer": "B",
            "model_raw_rationale": "Clinical reasoning.",
            "rationale": "Clinical reasoning.",
            "answer": "B",
            "answer_text": "Beta",
            "answer_correct": True,
            "canonical_response": response,
            "retrieval_queries": {
                "question_only": "Question?",
                "rationale_only": "Clinical reasoning.",
                "rationale_answer": "Clinical reasoning.\n\nFinal answer: (B) Beta",
            },
            "quality_flags": [],
            "valid_for_layer_analysis": True,
            "rationale_finish_reason": "stop",
            "rationale_stop_reason": "### END OF REASONING ###",
            "rationale_token_ids": [1, 2],
            "rationale_stats": {"token_count": 2, "ppl": 1.5},
            "choice_token_id": 2,
            "choice_logprobs": {"A": -2.0, "B": -0.1, "C": -3.0, "D": -4.0},
            "user_prompt_sha256": "a",
            "rendered_rationale_prompt_sha256": "b",
        }
        source = {"row_idx": 1, "split": "train", "subject": "test"}
        row = artifact_row(trace, source)
        self.assertEqual(row["retrieval_query"], "Clinical reasoning.\n\nFinal answer: (B) Beta")
        self.assertEqual(row["parsed"]["final_answer"], "B")
        self.assertTrue(row["parsed"]["final_answer_correct"])

        protocol = {"prompt_profile": "paper_compatible_three_anchor"}
        query, answer = resolve_rationale_query(row, protocol)
        self.assertEqual(query, row["retrieval_query"])
        self.assertEqual(answer, "B")
        self.assertEqual(retrieval_query_contract(protocol)["query_field"], "retrieval_query")

    def test_feature_complete_marker_checks_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "meta": root / "questions.jsonl",
                "tensor": root / "features.safetensors",
                "complete": root / "COMPLETE.json",
            }
            paths["meta"].write_text("{}\n", encoding="utf-8")
            paths["tensor"].write_bytes(b"placeholder")
            paths["complete"].write_text(
                json.dumps({"run_version": FEATURE_RUN_VERSION, "question_count": 1, "layers": [4, 28]}),
                encoding="utf-8",
            )
            self.assertTrue(complete_valid(paths, 1, [4, 28]))
            self.assertFalse(complete_valid(paths, 1, [4, 20]))


if __name__ == "__main__":
    unittest.main()

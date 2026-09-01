from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_rag2_anchored_semantic_subset_traces import (  # noqa: E402
    RUN_VERSION as GENERATION_RUN_VERSION,
    compact_trace,
    normalized_subset_rows,
    render_subset_document_text,
    sha256_file,
    shard_paths,
    valid_complete,
)
from materialize_rag2_semantic_subset_plan import (  # noqa: E402
    make_question_plan,
    policy_masks,
)


def candidate_row(labels: list[str] | None = None) -> dict:
    del labels
    sample_id = "medqa:train:000001"
    return {
        "sample_id": sample_id,
        "row_idx": 1,
        "dataset": "medqa",
        "split": "train",
        "question": "Which option is correct?",
        "options": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
        "answer": "B",
        "candidate_documents": [
            {
                "rerank_rank": rank,
                "stable_id": f"doc::{rank}",
                "source": "pubmed",
                "text": f"Evidence body {rank}.",
            }
            for rank in range(1, 9)
        ],
    }


def semantic_rows(labels: list[str]) -> list[dict]:
    sample_id = "medqa:train:000001"
    return [
        {
            "dataset": "medqa",
            "sample_id": sample_id,
            "doc_rank": rank,
            "doc_stable_id": f"doc::{rank}",
            "pair_id": f"{sample_id}::{rank}::doc::{rank}",
            "semantic_label": label,
            "confidence": 0.9,
        }
        for rank, label in enumerate(labels, start=1)
    ]


class SemanticSubsetTraceTests(unittest.TestCase):
    def test_policy_masks_preserve_rerank_order_and_exclude_mixed(self) -> None:
        labels = [
            "direct_support",
            "supporting_evidence",
            "no_evidence",
            "misleading_evidence",
            "indeterminate_or_mixed",
            "no_evidence",
            "supporting_evidence",
            "direct_support",
        ]
        masks = policy_masks(labels)
        self.assertEqual(masks["all_top8"], 0xFF)
        self.assertEqual(masks["direct_all"], (1 << 0) | (1 << 7))
        self.assertEqual(masks["supporting_all"], (1 << 1) | (1 << 6))
        self.assertEqual(masks["semantic_valid_all"], 0b11000011)
        self.assertEqual(masks["semantic_invalid_all"], 0b00101100)
        self.assertFalse(masks["semantic_valid_all"] & (1 << 4))
        self.assertFalse(masks["semantic_invalid_all"] & (1 << 4))
        self.assertEqual(masks["valid_pair"], 0b00000011)
        self.assertEqual(masks["valid_pair_plus_no"], 0b00000101)
        self.assertEqual(masks["valid_pair_plus_misleading"], 0b00001001)

    def test_plan_deduplicates_identical_policy_masks(self) -> None:
        labels = ["direct_support"] * 8
        plan, stats = make_question_plan(
            candidate_row(labels),
            semantic_rows(labels),
            dataset="medqa",
            source_split="train",
            analysis_split="val",
            top_k=8,
        )
        self.assertEqual(stats["generated_subsets"], 2)
        by_mask = {row["mask"]: row for row in plan["generation_subsets"]}
        self.assertEqual(
            by_mask[0xFF]["policies"],
            ["all_top8", "direct_all", "semantic_valid_all"],
        )
        self.assertEqual(by_mask[0x03]["policies"], ["valid_pair"])
        self.assertEqual(
            plan["policy_assignments"]["supporting_all"]["status"],
            "unavailable",
        )
        self.assertEqual(plan["analysis_split"], "val")

    def test_generator_reads_exact_plan_schema_and_renders_bodies_only(self) -> None:
        labels = [
            "direct_support",
            "supporting_evidence",
            "no_evidence",
            "misleading_evidence",
            "no_evidence",
            "supporting_evidence",
            "direct_support",
            "no_evidence",
        ]
        plan, _ = make_question_plan(
            candidate_row(labels),
            semantic_rows(labels),
            dataset="medqa",
            source_split="train",
            analysis_split="test",
            top_k=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset_plan.jsonl"
            path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            normalized = list(normalized_subset_rows(path, "medqa", "train", 8))
        self.assertEqual(len(normalized), 1)
        self.assertEqual(len(normalized[0]["subsets"]), len(plan["generation_subsets"]))

        documents = []
        for rank, source in enumerate(candidate_row()["candidate_documents"], start=1):
            documents.append(
                {
                    **source,
                    "pair_id": f"medqa:train:000001::{rank}::doc::{rank}",
                }
            )
        candidate = {"documents": documents}
        subset = {"document_ranks": [2, 5, 8]}
        text, metadata = render_subset_document_text(candidate, subset, 0)
        self.assertEqual(text, "Evidence body 2.\n\nEvidence body 5.\n\nEvidence body 8.")
        self.assertEqual([row["rerank_rank"] for row in metadata], [2, 5, 8])
        self.assertNotIn("supporting_evidence", text)
        self.assertNotIn("pubmed", text)

    def test_compact_trace_flags_incomplete_choice_distribution(self) -> None:
        metadata = {
            "dataset": "medqa",
            "split": "train",
            "sample_id": "q1",
            "row_idx": 0,
            "subset_id": "q1::subset::03",
            "document_mask": 3,
            "document_ranks": [1, 2],
            "document_count": 2,
            "policies": ["valid_pair"],
            "semantic_labels": ["direct_support", "supporting_evidence"],
            "semantic_counts": {},
            "documents": [],
            "document_text_sha256": "abc",
        }
        trace = {
            "quality_flags": [],
            "choice_logprobs": {"A": -2.0, "B": None, "C": -1.0, "D": -4.0},
            "answer_correct": True,
        }
        compact = compact_trace(trace, metadata)
        self.assertFalse(compact["valid_for_subset_analysis"])
        self.assertEqual(compact["choice_logprob_invalid_labels"], ["B"])
        self.assertIn("missing_or_nonfinite_choice_logprobs", compact["quality_flags"])

    def test_complete_marker_is_contract_and_content_keyed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = shard_paths(Path(directory), "medqa", "train", 0)
            paths["root"].mkdir(parents=True)
            paths["rows"].write_text("{}\n", encoding="utf-8")
            marker = {
                "run_version": GENERATION_RUN_VERSION,
                "contract_fingerprint": "contract",
                "question_count": 1,
                "subset_count": 1,
                "sample_ids_sha256": "samples",
                "rows_size_bytes": paths["rows"].stat().st_size,
                "rows_sha256": sha256_file(paths["rows"]),
            }
            paths["complete"].write_text(json.dumps(marker), encoding="utf-8")
            self.assertTrue(
                valid_complete(
                    paths,
                    fingerprint="contract",
                    expected_questions=1,
                    expected_subsets=1,
                    expected_sample_hash="samples",
                )
            )
            paths["rows"].write_text("{\"changed\": true}\n", encoding="utf-8")
            self.assertFalse(
                valid_complete(
                    paths,
                    fingerprint="contract",
                    expected_questions=1,
                    expected_subsets=1,
                    expected_sample_hash="samples",
                )
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_rag2_anchored_document_features import (  # noqa: E402
    estimate_storage,
    feature_paths,
    process_shard,
    validate_no_rag_contract,
)
from generate_rag2_anchored_document_traces import (  # noqa: E402
    RUN_VERSION,
    compact_trace,
    load_candidate_contract,
    normalized_candidate_rows,
    resumable_complete_pair_count,
    shard_paths,
    valid_complete,
)
from prepare_rag2_anchored_dynamic_oracle_candidates import project_union  # noqa: E402


class AnchoredDocumentPipelineTests(unittest.TestCase):
    def _candidate_fixture(self, root: Path) -> Namespace:
        candidate_root = root / "candidates"
        target = candidate_root / "medqa" / "train"
        target.mkdir(parents=True)
        documents = [
            {
                "rerank_rank": rank,
                "stable_id": f"pubmed::{rank}",
                "source": "pubmed",
                "local_id": rank,
                "text": f"Evidence sentence {rank}.",
            }
            for rank in range(1, 9)
        ]
        row = {
            "sample_id": "medqa:train:000001",
            "row_idx": 1,
            "question": "Question?",
            "options": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
            "answer": "B",
            "candidate_documents": documents,
        }
        (target / "candidates_top8.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        (target / "candidate_manifest.json").write_text(
            json.dumps(
                {
                    "type": "rag2_filter_candidate_dataset",
                    "dataset": "medqa",
                    "split": "train",
                    "top_k": 8,
                    "candidate_layout": "source_balanced",
                    "selected_question_count": 1,
                    "candidate_pool_top_k": 32,
                    "per_source_top_k": 8,
                }
            ),
            encoding="utf-8",
        )
        return Namespace(
            candidate_root=candidate_root,
            candidate_file="candidates_top8.jsonl",
            split="train",
            docs_per_question=8,
            allow_variable_docs_per_question=False,
        )

    def test_candidate_contract_and_pair_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._candidate_fixture(root)
            contract = load_candidate_contract(args, "medqa")
            self.assertEqual(contract["question_count"], 1)
            self.assertEqual(contract["pair_count"], 8)
            rows = list(
                normalized_candidate_rows(
                    Path(contract["candidate_path"]), "medqa", "train", 8
                )
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["documents"]), 8)
            pair_ids = [document["pair_id"] for document in rows[0]["documents"]]
            self.assertEqual(len(set(pair_ids)), 8)
            self.assertTrue(pair_ids[0].startswith("medqa:train:000001::1::"))

    def test_compact_trace_and_atomic_completion_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = shard_paths(root, "medqa", "train", 0)
            paths["root"].mkdir(parents=True)
            source = {"row_idx": 4, "split": "train"}
            trace = {
                "pilot_version": "pilot",
                "document": {
                    "pair_id": "sample::1::doc",
                    "rerank_rank": 1,
                    "text": "Evidence",
                },
            }
            compact = compact_trace(trace, source)
            self.assertNotIn("pilot_version", compact)
            self.assertNotIn("text", compact["document"])
            self.assertEqual(compact["pair_id"], "sample::1::doc")
            self.assertEqual(compact["split"], "train")
            paths["pairs"].write_text("{}\n", encoding="utf-8")
            paths["complete"].write_text(
                json.dumps(
                    {
                        "run_version": RUN_VERSION,
                        "question_count": 1,
                        "pair_count": 8,
                        "pairs_size_bytes": paths["pairs"].stat().st_size,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(valid_complete(paths, 1, 8))
            self.assertEqual(resumable_complete_pair_count(paths, 1), 8)

    def test_variable_candidate_contract_and_dynamic_topk_union(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_root = root / "candidates"
            target = candidate_root / "medqa" / "test"
            target.mkdir(parents=True)
            sources = ["pubmed", "pmc", "cpg", "textbooks"]
            initial = []
            for source in sources:
                for rank in range(1, 5):
                    initial.append(
                        {
                            "source": source,
                            "stable_id": f"{source}:{rank}",
                            "text": f"{source} evidence {rank}",
                            "retrieval_score": float(100 - rank),
                        }
                    )
            reranked = [dict(document) for document in reversed(initial)]
            for index, document in enumerate(reranked, 1):
                document["rerank_rank"] = index
                document["rerank_score"] = float(100 - index)
            union, selected = project_union(
                {
                    "key": "medqa::test::q1::0",
                    "initial_documents": initial,
                    "reranked_documents": reranked,
                },
                sources=sources,
                top_k_values=[1, 2, 4],
                master_per_source_top_k=4,
            )
            self.assertEqual({len(selected[str(k)]) for k in (1, 2, 4)}, {1, 2, 4})
            self.assertGreaterEqual(len(union), 4)
            self.assertEqual([row["rerank_rank"] for row in union], list(range(1, len(union) + 1)))
            row = {
                "sample_id": "medqa:test:q1",
                "row_idx": 0,
                "question": "Question?",
                "options": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
                "answer": "B",
                "candidate_documents": union,
            }
            (target / "candidates_topk_union.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (target / "candidate_manifest.json").write_text(
                json.dumps(
                    {
                        "type": "rag2_filter_candidate_dataset",
                        "dataset": "medqa",
                        "split": "test",
                        "candidate_layout": "source_balanced",
                        "selected_question_count": 1,
                        "selected_pair_count": len(union),
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                candidate_root=candidate_root,
                candidate_file="candidates_topk_union.jsonl",
                split="test",
                docs_per_question=1,
                allow_variable_docs_per_question=True,
            )
            contract = load_candidate_contract(args, "medqa")
            self.assertEqual(contract["pair_count"], len(union))
            normalized = list(
                normalized_candidate_rows(
                    Path(contract["candidate_path"]), "medqa", "test", 1, True
                )
            )
            self.assertEqual(len(normalized[0]["documents"]), len(union))

    def test_feature_extraction_accepts_legacy_trace_without_split(self) -> None:
        class FakeEncoding:
            input_ids = [1, 2, 3]
            anchor_indices = {"pre_rationale": 0, "post_rationale": 1, "pre_choice": 2}
            anchor_token_ids = {"pre_rationale": 1, "post_rationale": 2, "pre_choice": 3}
            anchor_token_text = {"pre_rationale": "a", "post_rationale": "b", "pre_choice": "c"}

        class FakeExtractor:
            layer_names = ["block_04"]

            def extract(self, rows):
                count = len(rows)
                probabilities = torch.tensor([[0.1, 0.7, 0.1, 0.1]]).repeat(count, 1)
                return (
                    {
                        "anchor_hidden": torch.zeros(count, 1, 3, 4),
                        "choice_logits": torch.log(probabilities),
                        "choice_probabilities": probabilities,
                    },
                    [FakeEncoding() for _ in rows],
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "legacy_pairs.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "trace_version": "rag2_anchored_rationale_answer_v1",
                        "dataset": "medqa",
                        "sample_id": "medqa:train:000001",
                        "pair_id": "medqa:train:000001::1::doc",
                        "row_idx": 1,
                        "doc_rank": 1,
                        "document": {"source": "pubmed", "stable_id": "doc"},
                        "document_text_used": "Evidence.",
                        "gold_answer": "B",
                        "answer": "B",
                        "answer_correct": True,
                        "rationale_stats": {"ppl": 1.2},
                        "quality_flags": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(batch_size=8, layers=[4], split="train")
            paths = feature_paths(root, "medqa", "train", "shard_00000")
            self.assertEqual(process_shard(args, FakeExtractor(), trace_path, paths), 1)
            metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["split"], "train")

    def test_no_rag_layer_contract_and_storage_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_rag = root / "no_rag"
            no_rag.mkdir()
            (no_rag / "feature_manifest.json").write_text(
                json.dumps(
                    {
                        "run_version": "rag2_anchored_no_rag_selected_layer_features_v1",
                        "trace_version": "rag2_paper_compatible_three_anchor_v1",
                        "prompt_version": "rag2_paper_compatible_three_anchor_prompt_v1",
                        "model_name_or_path": str((root / "model").resolve()),
                        "layers": [4, 12, 20, 28, 31],
                        "hidden_size": 4096,
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                no_rag_feature_root=no_rag,
                layers=[4, 28, 31],
                output_root=root,
                model_name_or_path=root / "model",
                minimum_free_space_gib=0.0,
            )
            manifest = validate_no_rag_contract(args)
            self.assertEqual(manifest["hidden_size"], 4096)
            estimate = estimate_storage(args, total_pairs=10, hidden_size=4096)
            self.assertGreater(estimate["estimated_tensor_gib"], 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_rag2_anchored_document_features import (  # noqa: E402
    estimate_storage,
    validate_no_rag_contract,
)
from generate_rag2_anchored_document_traces import (  # noqa: E402
    RUN_VERSION,
    compact_trace,
    load_candidate_contract,
    normalized_candidate_rows,
    shard_paths,
    valid_complete,
)


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
            source = {"row_idx": 4}
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

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.materialize_rag2_external_oracle_semantic_candidates import export_oracle


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class ExternalSemanticOracleExportTests(unittest.TestCase):
    def test_export_joins_semantic_labels_to_exact_evaluator_sample_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_root = root / "prepared"
            label_root = root / "labels"
            output_path = root / "semantic_oracle.jsonl"
            prepared_manifest = {
                "status": "complete",
                "preparation_version": "rag2_external_oracle_top32_semantic_candidates_v1",
                "transport_contract": {"documents_per_transport_row": 2},
                "datasets": {
                    "medqa": {"questions": 1, "pairs": 2, "transport_rows": 1}
                },
            }
            write_json(prepared_root / "prepare_manifest.json", prepared_manifest)
            sample_key = "medqa::test::medqa:test:000000::0"
            prepared_row = {
                "dataset": "medqa",
                "sample_id": "medqa:test:000000",
                "candidate_cache_key": sample_key,
                "candidate_documents": [
                    {"rerank_rank": 1, "stable_id": "doc-1"},
                    {"rerank_rank": 2, "stable_id": "doc-2"},
                ],
            }
            write_json(prepared_root / "candidates" / "medqa.jsonl", prepared_row)
            label_manifest = {
                "status": "complete",
                "annotation_version": "rag2_codex_evidence_utility_label_v2",
                "prompt_version": "rag2_codex_evidence_utility_prompt_v3_compact_item_index",
                "docs_per_question": 2,
                "allow_fewer_documents": False,
                "questions_per_batch": 10,
                "max_doc_chars": 0,
                "codex_bin": "codex",
                "codex_model_request": "gpt-5.6-terra",
                "codex_reasoning_effort": "medium",
                "web_search_enabled": False,
                "worker_count": 8,
                "datasets": {"medqa": {"questions": 1, "pairs": 2}},
            }
            write_json(label_root / "manifest.json", label_manifest)
            write_json(
                label_root / "external_oracle_top32_verification_report.json",
                {"status": "complete", "questions": 1, "pairs": 2},
            )
            label_path = label_root / "medqa" / "codex_semantic_labels.jsonl"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("w", encoding="utf-8") as handle:
                for rank, (stable_id, label) in enumerate(
                    (("doc-1", "direct_support"), ("doc-2", "supporting_evidence")),
                    start=1,
                ):
                    handle.write(
                        json.dumps(
                            {
                                "dataset": "medqa",
                                "sample_id": "medqa:test:000000",
                                "doc_rank": rank,
                                "doc_stable_id": stable_id,
                                "pair_id": f"medqa:test:000000::{rank}::{stable_id}",
                                "semantic_label": label,
                                "confidence": 0.9,
                            }
                        )
                        + "\n"
                    )

            args = Namespace(
                prepared_root=prepared_root,
                label_root=label_root,
                output_path=output_path,
                datasets=["medqa"],
                resume=True,
            )
            manifest = export_oracle(args)
            self.assertEqual(manifest["questions"], 1)
            self.assertEqual(manifest["pairs"], 2)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_key"] for row in rows], [sample_key, sample_key])
            self.assertEqual(
                [row["semantic_label"] for row in rows],
                ["direct_support", "supporting_evidence"],
            )
            self.assertEqual(
                [row["doc_stable_id"] for row in rows],
                ["doc-1", "doc-2"],
            )

            resumed = export_oracle(args)
            self.assertEqual(resumed["input_fingerprint"], manifest["input_fingerprint"])


if __name__ == "__main__":
    unittest.main()

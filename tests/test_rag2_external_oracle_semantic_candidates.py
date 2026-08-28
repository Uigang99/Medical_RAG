from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_rag2_external_oracle_semantic_candidates.py"
SPEC = importlib.util.spec_from_file_location("materialize_rag2_external_oracle_semantic_candidates", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ExternalOracleSemanticCandidatesTest(unittest.TestCase):
    def test_top32_is_split_into_four_top8_transport_rows_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "benchmarks"
            benchmark_path = benchmark_root / "medqa" / "test.jsonl"
            sample_id = "medqa:test:000000"
            write_jsonl(
                benchmark_path,
                [
                    {
                        "id": sample_id,
                        "dataset": "medqa",
                        "split": "test",
                        "question": "What is the diagnosis?",
                        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
                        "answer": "B",
                        "answers": ["B"],
                    }
                ],
            )
            cache_path = root / "cache" / "candidates.jsonl"
            documents = [
                {
                    "rerank_rank": rank,
                    "stable_id": f"doc-{rank}",
                    "source": ("pubmed", "pmc", "cpg", "textbooks")[(rank - 1) % 4],
                    "title": f"title {rank}",
                    "text": f"evidence {rank}.",
                    "rerank_score": 1.0 / rank,
                }
                for rank in range(1, 33)
            ]
            write_jsonl(
                cache_path,
                [
                    {
                        "key": "medqa::test::0",
                        "dataset": "medqa",
                        "sample_id": sample_id,
                        "row_idx": 0,
                        "dense_query_mode": "rationale",
                        "reranked_documents": documents,
                    }
                ],
            )
            cache_manifest_path = cache_path.parent / "manifest.json"
            cache_manifest_path.write_text(
                json.dumps(
                    {
                        "type": "rag2_mcq_eval_candidates",
                        "rows": 1,
                        "candidate_layout": "source_balanced",
                        "rerank_top_k": 128,
                        "dense_query_mode": "rationale",
                        "prompt_profile": "paper_compatible_three_anchor",
                    }
                ),
                encoding="utf-8",
            )
            prepared_root = root / "prepared"
            original_expected = MODULE.EXPECTED_DATASET_QUESTIONS["medqa"]
            MODULE.EXPECTED_DATASET_QUESTIONS["medqa"] = 1
            try:
                manifest = MODULE.prepare(
                    SimpleNamespace(
                        candidate_cache_path=cache_path,
                        candidate_cache_manifest_path=cache_manifest_path,
                        benchmark_root=benchmark_root,
                        output_root=prepared_root,
                        datasets=["medqa"],
                        top_k=32,
                        documents_per_block=8,
                        resume=False,
                    )
                )
            finally:
                MODULE.EXPECTED_DATASET_QUESTIONS["medqa"] = original_expected
            rows = [json.loads(line) for line in (prepared_root / "candidates" / "medqa.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(manifest["totals"]["pairs"], 32)
            self.assertEqual(manifest["totals"]["transport_rows"], 4)
            self.assertEqual(
                [[document["rerank_rank"] for document in row["candidate_documents"]] for row in rows],
                [list(range(1, 9)), list(range(9, 17)), list(range(17, 25)), list(range(25, 33))],
            )

            label_root = root / "labels"
            label_manifest = {
                "status": "complete",
                "annotation_version": MODULE.ANNOTATION_VERSION,
                "prompt_version": MODULE.PROMPT_VERSION,
                "docs_per_question": 8,
                "allow_fewer_documents": False,
                "questions_per_batch": 10,
                "max_doc_chars": 0,
                "codex_bin": "codex",
                "codex_model_request": "gpt-5.6-terra",
                "codex_reasoning_effort": "medium",
                "web_search_enabled": False,
                "worker_count": 8,
                "datasets": {"medqa": {"questions": 4, "pairs": 32, "planned_batches": 1}},
            }
            (label_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (label_root / "manifest.json").write_text(json.dumps(label_manifest), encoding="utf-8")
            labels = [
                {
                    "pair_id": f"{sample_id}::{rank}::doc-{rank}",
                    "dataset": "medqa",
                    "sample_id": sample_id,
                    "doc_rank": rank,
                    "doc_stable_id": f"doc-{rank}",
                    "semantic_label": "direct_support" if rank == 1 else "no_evidence",
                }
                for rank in range(1, 33)
            ]
            write_jsonl(label_root / "medqa" / "codex_semantic_labels.jsonl", labels)
            report = MODULE.verify(
                SimpleNamespace(
                    prepared_root=prepared_root,
                    label_root=label_root,
                    datasets=["medqa"],
                    output_path=label_root / "verification.json",
                )
            )
            self.assertEqual(report["questions"], 1)
            self.assertEqual(report["pairs"], 32)
            self.assertEqual(report["datasets"]["medqa"]["label_distribution"]["no_evidence"], 31)


if __name__ == "__main__":
    unittest.main()

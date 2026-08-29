from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_rag2_external_dynamic_semantic_delta.py"
SPEC = importlib.util.spec_from_file_location("materialize_rag2_external_dynamic_semantic_delta", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def label_manifest(pairs: int, questions: int, allow_fewer: bool) -> dict:
    return {
        "status": "complete",
        "annotation_version": MODULE.ANNOTATION_VERSION,
        "prompt_version": MODULE.PROMPT_VERSION,
        "docs_per_question": 8,
        "allow_fewer_documents": allow_fewer,
        "questions_per_batch": 10,
        "max_doc_chars": 0,
        "codex_bin": "codex",
        "codex_model_request": "gpt-5.6-terra",
        "codex_reasoning_effort": "medium",
        "web_search_enabled": False,
        "worker_count": 8,
        "datasets": {"medqa": {"questions": questions, "pairs": pairs, "planned_batches": 1}},
    }


class ExternalDynamicSemanticDeltaTest(unittest.TestCase):
    def test_only_missing_pair_is_labeled_then_exact_union_is_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            union_root = root / "union"
            existing_root = root / "existing"
            prepared_root = root / "prepared"
            pending_root = root / "pending"
            final_root = root / "final"
            sample_id = "medqa:test:000000"
            documents = [
                {
                    "stable_id": f"doc-{rank}",
                    "source": "pubmed",
                    "title": f"title {rank}",
                    "text": f"evidence {rank}",
                    "rerank_rank": rank,
                    "oracle_union_rank": rank,
                    "master_rerank_rank": rank if rank <= 8 else 36,
                    "metadata": {
                        "oracle_dynamic_top_k_membership": [8] if rank == 9 else [32],
                        "oracle_dynamic_rerank_rank_by_top_k": {"8": 8} if rank == 9 else {"32": rank},
                    },
                }
                for rank in range(1, 10)
            ]
            candidate_row = {
                "dataset": "medqa",
                "sample_id": sample_id,
                "row_idx": 0,
                "split": "test",
                "question": "Question?",
                "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
                "answer": "B",
                "answers": ["B"],
                "candidate_documents": documents,
            }
            union_candidate_path = union_root / "medqa" / "test" / "candidates_topk_union.jsonl"
            write_jsonl(union_candidate_path, [candidate_row])
            union_manifest = {
                "type": "rag2_paper_balanced_dynamic_oracle_candidate_union",
                "questions": 1,
                "pairs": 9,
                "questions_by_dataset": {"medqa": 1},
                "pairs_by_dataset": {"medqa": 9},
                "dynamic_top_k_values": [1, 2, 4, 8, 16, 32],
            }
            (union_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (union_root / "manifest.json").write_text(json.dumps(union_manifest), encoding="utf-8")
            (existing_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (existing_root / "manifest.json").write_text(
                json.dumps(label_manifest(8, 1, False)), encoding="utf-8"
            )
            existing_labels = [
                {
                    "id": f"{sample_id}::{rank}::doc-{rank}",
                    "pair_id": f"{sample_id}::{rank}::doc-{rank}",
                    "dataset": "medqa",
                    "sample_id": sample_id,
                    "doc_rank": rank,
                    "source": "pubmed",
                    "doc_stable_id": f"doc-{rank}",
                    "title": f"title {rank}",
                    "semantic_label": "no_evidence",
                    "topic_relation": "related",
                    "confidence": 0.8,
                    "evidence_sentence_indices": [],
                    "short_reason": "No useful evidence.",
                }
                for rank in range(1, 9)
            ]
            write_jsonl(existing_root / "medqa" / "codex_semantic_labels.jsonl", existing_labels)

            prepared = MODULE.prepare(
                SimpleNamespace(
                    candidate_union_root=union_root,
                    existing_label_root=existing_root,
                    output_root=prepared_root,
                    datasets=["medqa"],
                    max_documents_per_block=8,
                    resume=False,
                )
            )
            self.assertEqual(prepared["totals"]["reused_pairs"], 8)
            self.assertEqual(prepared["totals"]["pending_pairs"], 1)
            pending_row = json.loads((prepared_root / "pending_candidates" / "medqa.jsonl").read_text())
            self.assertEqual([doc["stable_id"] for doc in pending_row["candidate_documents"]], ["doc-9"])

            (pending_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (pending_root / "manifest.json").write_text(
                json.dumps(label_manifest(1, 1, True)), encoding="utf-8"
            )
            new_label = {
                **existing_labels[0],
                "id": f"{sample_id}::9::doc-9",
                "pair_id": f"{sample_id}::9::doc-9",
                "doc_rank": 9,
                "doc_stable_id": "doc-9",
                "semantic_label": "direct_support",
            }
            write_jsonl(pending_root / "medqa" / "codex_semantic_labels.jsonl", [new_label])
            merged = MODULE.merge(
                SimpleNamespace(
                    prepared_root=prepared_root,
                    new_label_root=pending_root,
                    output_root=final_root,
                    datasets=["medqa"],
                    resume=False,
                )
            )
            self.assertEqual(merged["pairs"], 9)
            self.assertEqual(merged["reused_pairs"], 8)
            self.assertEqual(merged["new_pairs"], 1)
            rows = [
                json.loads(line)
                for line in (final_root / "medqa" / "codex_semantic_labels.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["doc_rank"] for row in rows], list(range(1, 10)))
            self.assertEqual(rows[-1]["dynamic_union_origin"], "new_dynamic_delta")


if __name__ == "__main__":
    unittest.main()

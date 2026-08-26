from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_rag2_incremental_semantic_labels.py"
SPEC = importlib.util.spec_from_file_location("materialize_rag2_incremental_semantic_labels", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def manifest() -> dict:
    return {
        "status": "complete",
        "annotation_version": MODULE.ANNOTATION_VERSION,
        "prompt_version": MODULE.PROMPT_VERSION,
        "codex_model_request": "gpt-5.6-terra",
        "codex_reasoning_effort": "medium",
        "web_search_enabled": False,
        "label_definitions": {},
        "topic_relation_definitions": {},
    }


class IncrementalSemanticLabelsTest(unittest.TestCase):
    def test_prepare_reuses_by_stable_id_and_merge_remaps_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.jsonl"
            existing_path = root / "existing.jsonl"
            old_manifest_path = root / "old_manifest.json"
            prepared_root = root / "prepared"
            candidate = {
                "dataset": "medqa",
                "sample_id": "medqa:train:1",
                "question": "Question?",
                "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
                "answers": ["B"],
                "answer": "B",
                "candidate_documents": [
                    {
                        "rerank_rank": 1,
                        "stable_id": "new-doc",
                        "source": "pmc",
                        "title": "new",
                        "text": "new evidence",
                    },
                    {
                        "rerank_rank": 2,
                        "stable_id": "old-doc",
                        "source": "cpg",
                        "title": "old moved",
                        "text": "old evidence",
                    },
                ],
            }
            old_label = {
                "id": "medqa:train:1::1::old-doc",
                "pair_id": "medqa:train:1::1::old-doc",
                "dataset": "medqa",
                "sample_id": "medqa:train:1",
                "doc_rank": 1,
                "source": "cpg",
                "doc_stable_id": "old-doc",
                "title": "old",
                "semantic_label": "direct_support",
                "topic_relation": "unclear",
                "confidence": 0.9,
                "evidence_sentence_indices": [0],
                "short_reason": "useful",
            }
            write_jsonl(candidate_path, [candidate])
            write_jsonl(existing_path, [old_label])
            old_manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
            MODULE.prepare(
                SimpleNamespace(
                    candidates_paths=[candidate_path],
                    existing_label_paths=[existing_path],
                    existing_manifest_path=old_manifest_path,
                    output_root=prepared_root,
                    docs_per_question=2,
                    sqlite_work_dir=root,
                    resume=False,
                )
            )
            prepared_manifest = json.loads((prepared_root / "prepare_manifest.json").read_text())
            self.assertEqual(prepared_manifest["totals"]["reused_pairs"], 1)
            self.assertEqual(prepared_manifest["totals"]["pending_pairs"], 1)
            reused = json.loads((prepared_root / "reused_labels" / "medqa.jsonl").read_text())
            self.assertEqual(reused["doc_rank"], 2)
            self.assertEqual(reused["pair_id"], "medqa:train:1::2::old-doc")
            pending = json.loads((prepared_root / "pending_candidates" / "medqa.jsonl").read_text())
            self.assertEqual([doc["stable_id"] for doc in pending["candidate_documents"]], ["new-doc"])

            new_root = root / "new_labels"
            (new_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (new_root / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
            new_label = dict(old_label)
            new_label.update(
                {
                    "id": "medqa:train:1::1::new-doc",
                    "pair_id": "medqa:train:1::1::new-doc",
                    "doc_rank": 1,
                    "doc_stable_id": "new-doc",
                    "semantic_label": "no_evidence",
                    "evidence_sentence_indices": [],
                    "short_reason": "not useful",
                }
            )
            write_jsonl(new_root / "medqa" / "codex_semantic_labels.jsonl", [new_label])
            final_root = root / "final"
            MODULE.merge(
                SimpleNamespace(
                    prepared_root=prepared_root,
                    new_label_root=new_root,
                    output_root=final_root,
                    sqlite_work_dir=root,
                    resume=False,
                )
            )
            final_rows = [json.loads(line) for line in (final_root / "medqa" / "codex_semantic_labels.jsonl").read_text().splitlines()]
            self.assertEqual([row["doc_stable_id"] for row in final_rows], ["new-doc", "old-doc"])
            self.assertEqual([row["incremental_origin"] for row in final_rows], ["new_terra_medium", "reused_existing"])


if __name__ == "__main__":
    unittest.main()

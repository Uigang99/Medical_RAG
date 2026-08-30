from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_rag2_top8_baseline_rationales import generation_manifest_matches
from scripts.prepare_rag2_semantic_attention_controller import (
    resume_contract_differs_only_by_rationale_manifest_refresh,
)


class SemanticAttentionCacheResumeTest(unittest.TestCase):
    def test_complete_generation_manifest_ignores_only_completion_time(self) -> None:
        contract = {"run_version": "v1", "contract_fingerprint": "abc", "question_count": 3}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "generation_manifest.json"
            path.write_text(
                json.dumps({**contract, "completed_at": "first", "rationale_shards": 2}),
                encoding="utf-8",
            )
            self.assertTrue(generation_manifest_matches(path, contract, 2))
            self.assertFalse(generation_manifest_matches(path, contract, 3))

    def test_prepared_resume_accepts_only_manifest_metadata_refresh(self) -> None:
        previous = {
            "run_version": "features-v1",
            "rationale_cache": {
                "manifest": {
                    "path": "/cache/generation_manifest.json",
                    "size": 100,
                    "mtime_ns": 1,
                    "sha256": "old",
                },
                "contract_fingerprint": "rationale-fingerprint",
                "questions": 3000,
                "shards": 24,
            },
            "semantic_model_bundle": [{"path": "/model", "size": 10}],
            "contract_fingerprint": "feature-old",
        }
        current = copy_contract(previous)
        current["rationale_cache"]["manifest"].update(
            {"size": 101, "mtime_ns": 2, "sha256": "new"}
        )
        current["contract_fingerprint"] = "feature-new"
        self.assertTrue(
            resume_contract_differs_only_by_rationale_manifest_refresh(previous, current)
        )

        current["semantic_model_bundle"][0]["size"] = 11
        self.assertFalse(
            resume_contract_differs_only_by_rationale_manifest_refresh(previous, current)
        )


def copy_contract(value: dict) -> dict:
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()

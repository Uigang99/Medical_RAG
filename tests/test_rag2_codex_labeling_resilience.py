from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "label_rag2_candidates_with_codex.py"
SPEC = importlib.util.spec_from_file_location("label_rag2_candidates_with_codex", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Rag2CodexLabelingResilienceTest(unittest.TestCase):
    def test_capacity_error_is_extracted_from_jsonl_stdout(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t1"}),
                json.dumps(
                    {
                        "type": "error",
                        "message": "Selected model is at capacity. Please try a different model.",
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {"message": "Selected model is at capacity. Please try a different model."},
                    }
                ),
            ]
        )
        summary = MODULE.codex_failure_summary(
            1,
            stdout,
            "Reading additional input from stdin...\n",
        )
        self.assertIn("Selected model is at capacity", summary)
        self.assertNotIn("Reading additional input", summary)

    def test_retry_jitter_is_stable_and_bounded(self) -> None:
        args = type(
            "Args",
            (),
            {"retry_backoff_seconds": 60.0, "retry_jitter_fraction": 0.25},
        )()
        first = MODULE.retry_delay_seconds(args, "medqa", 42, 2)
        second = MODULE.retry_delay_seconds(args, "medqa", 42, 2)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 120.0)
        self.assertLessEqual(first, 150.0)


if __name__ == "__main__":
    unittest.main()

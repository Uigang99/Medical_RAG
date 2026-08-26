from __future__ import annotations

import sys
import json
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from datasets import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_rag2_anchored_paper_labels import (  # noqa: E402
    EXPECTED_GENERATION_POLICY,
    EXPECTED_PPL_SCOPE,
    EXPECTED_PROMPT_VERSION,
    EXPECTED_TRACE_VERSION,
    LABEL_DISCARD,
    LABEL_HELPFUL,
    LABEL_NOT_HELPFUL,
    assign_label,
    document_failures,
    no_rag_failures,
    rationale_ppl,
    main,
)
from train_rag2_filter_model_paper import tokenize_split  # noqa: E402
from medrag.filtering.rag2_official import resolve_configured_label_token_ids  # noqa: E402


def base_trace(**overrides):
    row = {
        "trace_version": EXPECTED_TRACE_VERSION,
        "prompt_version": EXPECTED_PROMPT_VERSION,
        "ppl_scope_version": EXPECTED_PPL_SCOPE,
        "generation_policy_version": EXPECTED_GENERATION_POLICY,
        "answer": "B",
        "gold_answer": "B",
        "rationale_stats": {"token_count": 10, "ppl": 2.0},
    }
    row.update(overrides)
    return row


class AnchoredPaperLabelTests(unittest.TestCase):
    def test_checkpoint_config_selects_three_class_and_historical_binary_contracts(self) -> None:
        class FakeTokenizer:
            unk_token_id = 0

            def encode(self, token, add_special_tokens=False):
                del add_special_tokens
                return {"[HELPFUL]": [10], "[NOT_HELPFUL]": [11], "[DISCARD]": [12]}[token]

        class ThreeClassConfig:
            rag2_filter_label_names = ["helpful", "not helpful", "discard"]
            rag2_filter_label_tokens = ["[HELPFUL]", "[NOT_HELPFUL]", "[DISCARD]"]

        names, token_ids = resolve_configured_label_token_ids(FakeTokenizer(), ThreeClassConfig())
        self.assertEqual(names, ("helpful", "not helpful", "discard"))
        self.assertEqual(token_ids, {"helpful": 10, "not helpful": 11, "discard": 12})

        class HistoricalBinaryConfig:
            rag2_filter_label_tokens = ["[HELPFUL]", "[NOT_HELPFUL]"]

        names, token_ids = resolve_configured_label_token_ids(FakeTokenizer(), HistoricalBinaryConfig())
        self.assertEqual(names, ("helpful", "not helpful"))
        self.assertEqual(token_ids, {"helpful": 10, "not helpful": 11})

    def test_exact_rag2_decision_table(self) -> None:
        tau = 0.4
        self.assertEqual(assign_label(False, True, -99.0, tau), (LABEL_HELPFUL, True))
        self.assertEqual(assign_label(True, False, -99.0, tau), (LABEL_NOT_HELPFUL, True))
        self.assertEqual(assign_label(True, True, 0.4, tau), (LABEL_HELPFUL, True))
        self.assertEqual(assign_label(True, True, 0.399, tau), (LABEL_DISCARD, False))
        self.assertEqual(assign_label(False, False, 0.4, tau), (LABEL_NOT_HELPFUL, True))
        self.assertEqual(assign_label(False, False, 0.399, tau), (LABEL_DISCARD, False))

    def test_no_rag_and_document_quality_contract(self) -> None:
        no_row = base_trace(valid=True, truncated_by_max_tokens=False)
        self.assertEqual(no_rag_failures(no_row), [])
        no_meta = {
            "gold_answer": "B",
            "trace_version": EXPECTED_TRACE_VERSION,
            "prompt_version": EXPECTED_PROMPT_VERSION,
            "ppl_scope_version": EXPECTED_PPL_SCOPE,
            "generation_policy_version": EXPECTED_GENERATION_POLICY,
        }
        doc_row = base_trace(
            valid_for_layer_analysis=True,
            document_text_used="Clinical evidence.",
        )
        self.assertEqual(document_failures(doc_row, no_meta), [])

    def test_rationale_ppl_requires_nonempty_generated_span(self) -> None:
        self.assertEqual(rationale_ppl(base_trace()), 2.0)
        self.assertIsNone(rationale_ppl(base_trace(rationale_stats={"token_count": 0, "ppl": 2.0})))
        self.assertIsNone(rationale_ppl(base_trace(rationale_stats={"token_count": 2, "ppl": float("nan")})))

    def test_end_to_end_question_split_tau_and_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_root = root / "no"
            doc_root = root / "doc"
            output_root = root / "out"
            three_class_output_root = root / "out_three_class"
            no_shard = no_root / "trace_shards/medqa/train/shard_00000"
            doc_shard = doc_root / "trace_shards/medqa/train/shard_00000"
            no_shard.mkdir(parents=True)
            doc_shard.mkdir(parents=True)
            no_rows = []
            doc_rows = []
            for index in range(10):
                sample_id = f"medqa:train:{index:06d}"
                no_rows.append(
                    {
                        **base_trace(valid=True, truncated_by_max_tokens=False),
                        "dataset": "medqa",
                        "sample_id": sample_id,
                    }
                )
                for rank, answer, ppl in ((1, "B", 1.0), (2, "A", 2.0), (3, "B", 2.0)):
                    doc_rows.append(
                        {
                            **base_trace(
                                answer=answer,
                                valid_for_layer_analysis=True,
                                rationale_stats={"token_count": 10, "ppl": ppl},
                            ),
                            "dataset": "medqa",
                            "sample_id": sample_id,
                            "pair_id": f"{sample_id}::{rank}::doc{rank}",
                            "row_idx": index,
                            "doc_rank": rank,
                            "question": "Which choice is correct?",
                            "options": {"A": "No", "B": "Yes", "C": "Maybe", "D": "Unknown"},
                            "document": {"source": "pubmed", "stable_id": f"doc{rank}"},
                            "document_text_used": f"Evidence {rank}.",
                        }
                    )
            (no_shard / "questions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in no_rows), encoding="utf-8"
            )
            (doc_shard / "pairs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in doc_rows), encoding="utf-8"
            )
            (no_root / "generation_manifest.json").write_text(
                json.dumps({"datasets": {"medqa": 10}}), encoding="utf-8"
            )
            (doc_root / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "datasets": {"medqa": 10},
                        "pairs_by_dataset": {"medqa": 30},
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "build",
                "--no-rag-root",
                str(no_root),
                "--document-trace-root",
                str(doc_root),
                "--output-root",
                str(output_root),
                "--datasets",
                "medqa",
                "--max-doc-rank",
                "3",
                "--no-show-progress",
            ]
            with patch.object(sys, "argv", argv):
                main()
            manifest = json.loads((output_root / "medqa/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["threshold_summary"]["tau"], 1.0)
            self.assertEqual(manifest["summary"]["splits"]["train"]["questions"], 8)
            self.assertEqual(manifest["summary"]["splits"]["val"]["questions"], 1)
            self.assertEqual(manifest["summary"]["splits"]["test"]["questions"], 1)
            with (output_root / "medqa/train.jsonl").open() as handle:
                self.assertEqual(sum(1 for _ in handle), 16)

            three_class_argv = [
                *argv[: argv.index("--output-root") + 1],
                str(three_class_output_root),
                *argv[argv.index("--output-root") + 2 :],
                "--training-label-mode",
                "three_class",
            ]
            with patch.object(sys, "argv", three_class_argv):
                main()
            three_class_manifest = json.loads(
                (three_class_output_root / "medqa/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(three_class_manifest["label_protocol"]["training_label_mode"], "three_class")
            self.assertEqual(
                three_class_manifest["label_protocol"]["training_labels"],
                ["Helpful", "Not Helpful", "Discard"],
            )
            with (three_class_output_root / "medqa/train.jsonl").open() as handle:
                three_class_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(three_class_rows), 24)
            self.assertEqual(
                {row["target"] for row in three_class_rows},
                {"helpful", "not helpful", "discard"},
            )

    def test_released_overflow_preprocessing_repeats_each_pair_target(self) -> None:
        class FakeTokenizer:
            def __call__(self, texts=None, *, text_target=None, return_overflowing_tokens=False, **kwargs):
                if text_target is not None:
                    return {"input_ids": [[10] if value == "[HELPFUL]" else [11] for value in text_target]}
                assert return_overflowing_tokens
                return {
                    "input_ids": [[1, 2], [2, 3], [4, 5]],
                    "attention_mask": [[1, 1], [1, 1], [1, 1]],
                    "overflow_to_sample_mapping": [0, 0, 1],
                }

        dataset = Dataset.from_dict(
            {
                "_filter_input": ["first", "second"],
                "target": ["helpful", "not helpful"],
            }
        )
        args = Namespace(
            overlength_policy="overflow",
            max_seq_length=4,
            doc_stride=1,
            max_target_length=4,
            preprocessing_num_workers=1,
        )
        result = tokenize_split(
            dataset,
            FakeTokenizer(),
            args,
            "fixture",
            {"helpful": "[HELPFUL]", "not helpful": "[NOT_HELPFUL]"},
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result["labels"], [[10], [10], [11]])


if __name__ == "__main__":
    unittest.main()

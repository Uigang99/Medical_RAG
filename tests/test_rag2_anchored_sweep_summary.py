from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_rag2_anchored_paper_reproduction_sweep import (
    condition_specs,
    render_table,
    summarize_rows,
)


class AnchoredSweepSummaryTests(unittest.TestCase):
    def test_summary_reports_micro_eight_macro_and_three_group_macro(self) -> None:
        expected_counts = {
            "medmcqa": 2,
            "medqa": 2,
            "mmlu_anatomy": 1,
            "mmlu_clinical_knowledge": 1,
            "mmlu_college_biology": 1,
            "mmlu_college_medicine": 1,
            "mmlu_medical_genetics": 1,
            "mmlu_professional_medicine": 1,
        }
        correctness = {
            "medmcqa": [True, False],
            "medqa": [True, True],
            "mmlu_anatomy": [True],
            "mmlu_clinical_knowledge": [False],
            "mmlu_college_biology": [True],
            "mmlu_college_medicine": [False],
            "mmlu_medical_genetics": [True],
            "mmlu_professional_medicine": [False],
        }
        rows = []
        for dataset, values in correctness.items():
            for index, correct in enumerate(values):
                rows.append(
                    {
                        "sample": {"dataset": dataset, "id": f"{dataset}-{index}"},
                        "evaluation": {"correct": correct},
                        "context_document_count": 2,
                    }
                )
        metrics = summarize_rows(rows, expected_counts=expected_counts)
        self.assertAlmostEqual(metrics["dataset_accuracy"]["medmcqa"], 0.5)
        self.assertAlmostEqual(metrics["dataset_accuracy"]["medqa"], 1.0)
        self.assertAlmostEqual(metrics["mmlu_pooled_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["micro_accuracy"], 0.6)
        self.assertAlmostEqual(metrics["macro_8_accuracy"], 0.5625)
        self.assertAlmostEqual(metrics["macro_3_accuracy"], 2.0 / 3.0)
        self.assertEqual(metrics["mean_context_documents"], 2.0)

    def test_rendered_table_contains_all_requested_metrics(self) -> None:
        dataset_accuracy = {
            "medmcqa": 0.5,
            "medqa": 0.6,
            "mmlu_anatomy": 0.7,
            "mmlu_clinical_knowledge": 0.7,
            "mmlu_college_biology": 0.7,
            "mmlu_college_medicine": 0.7,
            "mmlu_medical_genetics": 0.7,
            "mmlu_professional_medicine": 0.7,
        }
        table = render_table(
            {
                "conditions": [
                    {
                        "top_k": 2,
                        "filtering": "RAG2 filtering",
                        "metrics": {
                            "mean_context_documents": 1.25,
                            "dataset_accuracy": dataset_accuracy,
                            "mmlu_pooled_accuracy": 0.7,
                            "micro_accuracy": 0.6,
                            "macro_8_accuracy": 0.65,
                            "macro_3_accuracy": 0.6,
                        },
                    }
                ]
            }
        )
        self.assertIn("Macro Avg (8)", table)
        self.assertIn("Macro Avg (3 groups)", table)
        self.assertIn("MMLU pooled", table)
        self.assertIn("| 2 | RAG2 filtering | 1.25 |", table)

    def test_generic_oracle_condition_naming(self) -> None:
        specs = condition_specs(
            Path("/reference"),
            Path("/oracle"),
            oracle_case_prefix="oracle_rag_margin_utility",
            oracle_display_label="Margin utility Oracle",
        )
        oracle = [row for row in specs if row["case"] == "oracle_rag"]
        self.assertEqual(len(oracle), 6)
        self.assertEqual(oracle[0]["filtering"], "Margin utility Oracle")
        self.assertEqual(
            oracle[0]["case_root"],
            Path("/oracle/oracle_rag_margin_utility_top1"),
        )


if __name__ == "__main__":
    unittest.main()

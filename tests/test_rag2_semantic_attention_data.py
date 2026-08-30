from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medrag.training.rag2_semantic_attention_data import (
    DeterministicQuestionBatchSampler,
    RAG2SemanticAttentionDataset,
    SemanticAttentionDataSources,
    build_semantic_attention_index,
    make_semantic_attention_build_plan,
)


LABELS = (
    "direct_support",
    "supporting_evidence",
    "no_evidence",
    "misleading_evidence",
    "indeterminate_or_mixed",
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class SyntheticSemanticAttentionData:
    def __init__(self, root: Path, *, questions: int = 3, missing_last_label: bool = False) -> None:
        self.root = root
        self.dataset = "medqa"
        self.candidates = root / "candidates" / "medqa" / "train" / "candidates_top8.jsonl"
        self.labels = root / "labels" / "terra_medium" / "medqa" / "codex_semantic_labels.jsonl"
        self.no_rag = root / "no_rag" / "medqa" / "train" / "no_rag_generations.jsonl"
        self.split_root = root / "sample_ids"
        split_names = ("train", "val", "test")
        candidate_rows: list[dict] = []
        label_rows: list[dict] = []
        no_rag_rows: list[dict] = []
        assignments: dict[str, list[str]] = {split: [] for split in split_names}
        for question_index in range(questions):
            sample_id = f"medqa:train:{question_index:06d}"
            split = split_names[question_index % len(split_names)]
            assignments[split].append(sample_id)
            documents = []
            for rank in range(1, 9):
                stable_id = f"rag2::pmc::doc_{question_index}_{rank}"
                documents.append(
                    {
                        "rerank_rank": rank,
                        "stable_id": stable_id,
                        "source": "pmc",
                        "title": f"Title {rank}",
                        "text": f"Evidence text for question {question_index} document {rank}.",
                    }
                )
                label = LABELS[(rank - 1) % len(LABELS)]
                label_rows.append(
                    {
                        "dataset": self.dataset,
                        "sample_id": sample_id,
                        "pair_id": f"{sample_id}::{rank}::{stable_id}",
                        "doc_rank": rank,
                        "doc_stable_id": stable_id,
                        "source": "pmc",
                        "semantic_label": label,
                        "confidence": 0.9,
                        "topic_relation": "related" if label == "no_evidence" else None,
                        "evidence_sentence_indices": [] if label == "no_evidence" else [1],
                        "short_reason": f"Reason {rank}",
                    }
                )
            candidate_rows.append(
                {
                    "dataset": self.dataset,
                    "split": "train",
                    "sample_id": sample_id,
                    "row_idx": question_index,
                    "question": f"Question {question_index}?",
                    "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                    "answer": "B" if question_index % 2 == 0 else "A",
                    "answers": ["B"],
                    "candidate_documents": documents,
                }
            )
            no_rag_rows.append(
                {
                    "dataset": self.dataset,
                    "split": "train",
                    "sample_id": sample_id,
                    "gold_answer": "B",
                    "answer": "B",
                    "answer_correct": question_index % 2 == 0,
                    "valid": True,
                    "canonical_generation": f"Rationale {question_index}\nFinal answer: (B) Beta",
                    "parsed": {
                        "final_answer": "B" if question_index % 2 == 0 else "A",
                        "final_answer_correct": question_index % 2 == 0,
                    },
                    "choice_logprobs": {"A": -2.0, "B": -0.1, "C": -3.0, "D": -4.0},
                }
            )
        if missing_last_label:
            label_rows.pop()
        write_jsonl(self.candidates, candidate_rows)
        write_jsonl(self.labels, label_rows)
        write_jsonl(self.no_rag, no_rag_rows)
        self.split_root.mkdir(parents=True, exist_ok=True)
        for split, sample_ids in assignments.items():
            (self.split_root / f"{split}.txt").write_text("\n".join(sample_ids) + "\n")

        (self.candidates.parent / "candidate_manifest.json").write_text(
            json.dumps({"selected_question_count": questions})
        )
        self.labels.parent.parent.joinpath("manifest.json").write_text(
            json.dumps(
                {
                    "datasets": {
                        self.dataset: {
                            "final_questions": questions,
                            "final_pairs": len(label_rows),
                        }
                    }
                }
            )
        )
        self.no_rag.parent.joinpath("manifest.json").write_text(json.dumps({"rows": questions}))

    def sources(self) -> SemanticAttentionDataSources:
        return SemanticAttentionDataSources(
            dataset=self.dataset,
            candidates_path=self.candidates,
            semantic_labels_path=self.labels,
            split_ids_root=self.split_root,
            no_rag_path=self.no_rag,
        )


class SemanticAttentionDataTest(unittest.TestCase):
    def test_build_keeps_all_documents_and_masks_only_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSemanticAttentionData(Path(temporary))
            index = Path(temporary) / "grouped.sqlite"
            result = build_semantic_attention_index(
                fixture.sources(), index, checkpoint_every=2, show_progress=False
            )
            self.assertFalse(result.reused)
            self.assertEqual(result.manifest["summary"]["questions"], 3)
            self.assertEqual(result.manifest["summary"]["documents"], 24)
            self.assertEqual(result.manifest["summary"]["semantic_loss_masked_documents"], 3)

            train = RAG2SemanticAttentionDataset(index, "train")
            self.assertEqual(len(train), 1)
            question = train[0]
            self.assertEqual(question.sample_id, "medqa:train:000000")
            self.assertEqual(question.gold_answers, ("B",))
            self.assertEqual(question.no_rag.predicted_answer, "B")
            self.assertEqual(len(question.documents), 8)
            self.assertEqual(tuple(document.rank for document in question.documents), tuple(range(1, 9)))
            mixed = question.documents[4]
            self.assertEqual(mixed.semantic_label, "indeterminate_or_mixed")
            self.assertFalse(mixed.semantic_loss_mask)
            self.assertIsNone(mixed.semantic_class_id)
            self.assertIsNone(mixed.semantic_support_target)
            self.assertTrue(question.documents[0].semantic_loss_mask)
            self.assertEqual(question.documents[0].semantic_support_target, 1)
            self.assertEqual(question.documents[2].semantic_support_target, 0)
            train.close()

            reused = build_semantic_attention_index(fixture.sources(), index, show_progress=False)
            self.assertTrue(reused.reused)
            self.assertEqual(reused.manifest["source_fingerprint"], result.manifest["source_fingerprint"])

    def test_missing_raw_semantic_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSemanticAttentionData(
                Path(temporary), questions=1, missing_last_label=True
            )
            # Remove the intentionally inconsistent annotation manifest count so
            # the failure is exercised at the exact candidate-pair join.
            fixture.labels.parent.parent.joinpath("manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing raw semantic label"):
                build_semantic_attention_index(
                    fixture.sources(), Path(temporary) / "missing.sqlite", show_progress=False
                )

    def test_split_overlap_is_rejected_before_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSemanticAttentionData(Path(temporary))
            duplicate = "medqa:train:000000\n"
            with (fixture.split_root / "val.txt").open("a", encoding="utf-8") as handle:
                handle.write(duplicate)
            with self.assertRaisesRegex(ValueError, "split overlap"):
                make_semantic_attention_build_plan(fixture.sources())

    def test_deterministic_batches_and_resume_cursor(self) -> None:
        first = DeterministicQuestionBatchSampler(10, 3, seed=17, epoch=2)
        second = DeterministicQuestionBatchSampler(10, 3, seed=17, epoch=2)
        batches = list(first)
        self.assertEqual(batches, list(second))
        self.assertEqual(len(batches), 4)
        self.assertEqual(sorted(item for batch in batches for item in batch), list(range(10)))

        resumed = DeterministicQuestionBatchSampler(10, 3, seed=17)
        resumed.load_state_dict(first.state_dict(next_batch=2))
        self.assertEqual(list(resumed), batches[2:])
        resumed.set_epoch(3)
        self.assertNotEqual(list(resumed), batches)


if __name__ == "__main__":
    unittest.main()

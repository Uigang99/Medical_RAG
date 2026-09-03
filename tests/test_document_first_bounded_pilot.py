import unittest

import numpy as np

from scripts.materialize_rag2_document_first_bounded_pilot import (
    LowestHashes,
    donor_map,
    jsd,
    semantic_negative,
    semantic_positive,
)


class DocumentFirstBoundedPilotTest(unittest.TestCase):
    def test_lowest_hash_reservoir_is_bounded_and_deterministic(self):
        reservoir = LowestHashes(2)
        for priority, sample_id in ((9, "q9"), (2, "q2"), (5, "q5"), (1, "q1")):
            reservoir.add(priority, sample_id)
        self.assertEqual(reservoir.values(), ["q1", "q2"])

    def test_jsd_zero_and_positive(self):
        a = np.asarray([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
        b = np.asarray([0.1, 0.7, 0.1, 0.1], dtype=np.float32)
        self.assertAlmostEqual(jsd(a, a), 0.0, places=7)
        self.assertGreater(jsd(a, b), 0.0)

    def test_semantic_groups(self):
        rows = [
            {"document": {"semantic_label": "direct_support"}},
            {"document": {"semantic_label": "supporting_evidence"}},
            {"document": {"semantic_label": "no_evidence"}},
            {"document": {"semantic_label": "misleading_evidence"}},
        ]
        self.assertEqual(len(semantic_positive(rows)), 1)
        self.assertEqual(len(semantic_negative(rows)), 2)

    def test_cross_question_donor_never_crosses_split(self):
        rows = []
        for split in ("train", "val"):
            for index in range(2):
                rows.append(
                    {
                        "cohort_split": split,
                        "sample_id": f"{split}-{index}",
                        "documents": [
                            {
                                "document": {
                                    "semantic_label": "direct_support",
                                    "semantic_confidence": 1.0,
                                    "rerank_rank": 1,
                                    "source": "pmc",
                                    "document_text": "short support text",
                                    "pair_id": f"{split}-pair-{index}",
                                }
                            }
                        ],
                    }
                )
        donors = donor_map(rows, 42)
        for sample_id, donor in donors.items():
            self.assertEqual(donor["donor_split"], sample_id.split("-")[0])
            self.assertNotEqual(donor["donor_sample_id"], sample_id)


if __name__ == "__main__":
    unittest.main()

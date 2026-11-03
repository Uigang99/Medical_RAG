import argparse
import unittest

from scripts import run_medical_pced_100 as medical


class MedicalPCEDCohortTest(unittest.TestCase):
    def test_seeded_sample_is_stable_and_train_only(self):
        for name in medical.DATASETS:
            with self.subTest(dataset=name):
                args = argparse.Namespace(dataset=name, fetch_k=256, sample_seed=20261103)
                _, _, corpus, first = medical.selected_inputs(args)
                _, _, _, second = medical.selected_inputs(args)
                self.assertEqual(100, len(first))
                self.assertEqual([q["qid"] for q in first], [q["qid"] for q in second])
                self.assertEqual(100, len({q["qid"] for q in first}))
                self.assertTrue(all(q["metadata"]["split"] == "train" for q in first))
                self.assertGreaterEqual(len(corpus), 256)


if __name__ == "__main__":
    unittest.main()

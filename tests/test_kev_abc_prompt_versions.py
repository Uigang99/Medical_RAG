from __future__ import annotations

import unittest

from medrag import kev_abc_prompt_versions as versions


PAIR = {
    "pair_id": "test-pair",
    "P1": {"id": "p1", "document_id": "d1", "claim": "Drug A reduces symptom B."},
    "P2": {"id": "p2", "document_id": "d2", "claim": "Symptom B occurs in disease C."},
    "question": "SECRET_QUESTION",
    "answer": "SECRET_GOLD",
}


class PromptVersionsTests(unittest.TestCase):
    def test_all_versions_are_fixed_choice_and_hide_gold(self):
        for version in versions.VERSIONS:
            for graph, count in (("A", 1), ("B", 3), ("C", 18)):
                with self.subTest(version=version, graph=graph):
                    job = versions.make_job(PAIR, graph, version)
                    self.assertEqual(len(job["body"]["questions"]), count)
                    self.assertNotIn("SECRET", str(job["body"]))
                    self.assertEqual(job, versions.make_job(PAIR, graph, version))

    def test_composite_reuses_exact_family_prompt_bodies(self):
        for graph, source in (("A", "v6_minimal_pair_test"),
                              ("B", "v4_relation_boundaries"),
                              ("C", "v4_relation_boundaries")):
            with self.subTest(graph=graph):
                selected = versions.make_job(PAIR, graph, "v7_family_composite")
                original = versions.make_job(PAIR, graph, source)
                self.assertEqual(selected["body"], original["body"])
                self.assertNotEqual(selected["id"], original["id"])


if __name__ == "__main__":
    unittest.main()

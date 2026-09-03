import unittest

from scripts.evaluate_rag2_document_first_prompt_order import (
    build_document_first_user_prompt,
    equal_token_allocation,
)


class Sample:
    id = "q"
    question = "Question?"
    options = {"A": "One", "B": "Two", "C": "Three", "D": "Four"}


class DocumentFirstPromptOrderTest(unittest.TestCase):
    def test_document_precedes_question(self):
        prompt = build_document_first_user_prompt(Sample(), "EVIDENCE")
        self.assertLess(prompt.index("EVIDENCE"), prompt.index("Here is the question:"))

    def test_waterfill_respects_capacity_and_budget(self):
        values = equal_token_allocation([3, 10, 20], 18)
        self.assertEqual(sum(values), 18)
        self.assertTrue(all(value <= capacity for value, capacity in zip(values, [3, 10, 20])))


if __name__ == "__main__":
    unittest.main()

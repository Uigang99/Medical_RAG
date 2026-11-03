import unittest
from pathlib import Path

import numpy as np

from scripts import run_loft_pced_paper as paper
from scripts.evaluate_loft_corpus_in_context import (
    sparse_match, shuffled_top90, vanilla_prompt,
)


class CorpusInContextContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        model = Path(__file__).resolve().parents[2] / "models/Llama-3.1-8B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

    def test_prompt_matches_official_vanilla_role_layout(self):
        docs = ["(One) CONTENT: First.", "(Two) CONTENT: Second."]
        user = "\n\n".join(docs) + paper.QUESTION.format(question="Who?")
        expected = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": paper.SYSTEM},
             {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        ) + paper.ANSWER_PREFIX
        self.assertEqual(vanilla_prompt(self.tokenizer, "Who?", docs), expected)
        self.assertLess(expected.index("First."), expected.index("Second."))
        self.assertLess(expected.index("Second."), expected.index("query: Who?"))

    def test_shuffled_top90_is_seeded_and_keeps_every_candidate(self):
        ranking = {"qid": "q1", "top90": [{"pid": str(i)} for i in range(90)]}
        first = shuffled_top90(ranking, np.random.default_rng(125))
        second = shuffled_top90(ranking, np.random.default_rng(125))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {str(i) for i in range(90)})
        self.assertNotEqual(first, [str(i) for i in range(90)])

    def test_sparse_lexical_match(self):
        self.assertEqual(sparse_match({"1": 2.0, "2": 1.0},
                                      {"1": 3.0, "3": 8.0}), 6.0)
        self.assertEqual(sparse_match({"1": 2.0}, {"2": 3.0}), 0.0)


if __name__ == "__main__":
    unittest.main()

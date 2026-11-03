import unittest
from pathlib import Path

import torch

from scripts.evaluate_loft_pced_no_rag import make_batch, no_rag_ids


class LoftPcedNoRagTest(unittest.TestCase):
    def test_amateur_prompt_has_question_but_no_document(self):
        from transformers import AutoTokenizer
        model = Path(__file__).resolve().parents[2] / "models/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        prompt = tokenizer.decode(no_rag_ids(tokenizer, "Who wrote Hamlet?"))
        self.assertIn("query: Who wrote Hamlet?", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertNotIn("CONTENT:", prompt)

    def test_batch_left_pads_without_changing_tokens(self):
        ids, mask = make_batch([("a", {}, [1, 2]), ("b", {}, [3])], 0, "cpu")
        self.assertTrue(torch.equal(ids, torch.tensor([[1, 2], [0, 3]])))
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1], [0, 1]])))


if __name__ == "__main__":
    unittest.main()

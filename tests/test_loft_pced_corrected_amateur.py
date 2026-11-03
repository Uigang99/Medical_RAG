import unittest
from pathlib import Path
import sys

from scripts import run_loft_pced_paper as paper
from scripts.evaluate_loft_pced_corrected_amateur import no_rag_ids, prompt_inputs


class CorrectedAmateurPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        model = Path(__file__).resolve().parents[2] / "models/Llama-3.1-8B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                                       padding_side="left")

    def test_expert_stream_is_exactly_the_paper_prompt(self):
        original_prefixes, original_question = paper.prompt_inputs(
            self.tokenizer, "Who wrote Hamlet?", ["(T) CONTENT: P"])
        revised_prefixes, revised_questions = prompt_inputs(
            self.tokenizer, "Who wrote Hamlet?", ["(T) CONTENT: P"])
        self.assertEqual(revised_prefixes[1], original_prefixes[1])
        self.assertEqual(revised_questions[1], original_question)
        self.assertLessEqual(
            len(self.tokenizer(revised_questions[0], add_special_tokens=False)["input_ids"]),
            len(self.tokenizer(revised_questions[1], add_special_tokens=False)["input_ids"]))

    def test_amateur_stream_has_no_document_instruction(self):
        rendered = self.tokenizer.decode(no_rag_ids(self.tokenizer, "Who wrote Hamlet?"))
        self.assertIn("query: Who wrote Hamlet?", rendered)
        self.assertIn("using your own knowledge", rendered)
        self.assertNotIn("documents above", rendered)
        self.assertNotIn("documents", rendered)
        self.assertNotIn("CONTENT:", rendered)

    def test_official_decoder_accepts_distinct_amateur_question(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "cache/pced_official"))
        from src.methods.pced import prefill, decode_from_cache

        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=4, max_position_embeddings=128,
                             bos_token_id=1, eos_token_id=2, pad_token_id=0)
        model = LlamaForCausalLM(config).eval()
        context = {"input_ids": torch.tensor([[1, 3, 4], [1, 5, 6]]),
                   "attention_mask": torch.ones((2, 3), dtype=torch.long)}
        question = {"input_ids": torch.tensor([[0, 7, 8, 9], [10, 11, 12, 13]]),
                    "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])}
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        cache = prefill(model, context)
        ids, winners = decode_from_cache(model, tokenizer, cache, context, question,
            max_new_tokens=2, beta=0.5, retrieval_scores=torch.tensor([0.9]),
            gamma=2.5, return_winner_trace=True)
        self.assertGreaterEqual(ids.numel(), 1)
        self.assertEqual(len(winners), ids.numel())


if __name__ == "__main__":
    unittest.main()

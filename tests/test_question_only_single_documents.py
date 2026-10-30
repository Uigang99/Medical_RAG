import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_question_only_single_documents as target


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages) + "\nAssistant: "


class LLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append(prompts)
        outputs = []
        for prompt in prompts:
            chosen = 1 if "first evidence" in prompt else 2
            scores = {i: SimpleNamespace(logprob=0. if i == chosen else -1.) for i in range(1, 5)}
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[scores])]))
        return outputs


class IndependentDocumentTest(unittest.TestCase):
    def test_duplicate_document_runs_once_and_rank_is_hidden(self):
        sample = SimpleNamespace(id="q1", dataset="medqa", answer="A", question="Example?",
                                 options={x: x for x in "ABCD"})
        first = {"corpus_id": "d1", "source": "pubmed", "title": "", "text": "first evidence"}
        second = {"corpus_id": "d2", "source": "cpg", "title": "", "text": "second evidence"}
        docs = {arm: [first, second, first, second, first, second, first, second]
                for arm in target.ARMS}
        block = [(sample, docs)]
        llm = LLM()
        reference = [{"predictions": {"no_rag": "C"}}]
        rows, calls = target.score_block(Tokenizer(), llm, None,
                                        dict(zip("ABCD", range(1, 5))), block, reference, "hash")
        self.assertEqual(calls, 2)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(llm.calls[0]), 2)
        self.assertTrue(all("[1]" in p and "[2]" not in p for p in llm.calls[0]))
        self.assertEqual(rows[0]["predictions"][target.ARMS[0]], list("ABABABAB"))
        self.assertEqual(rows[0]["document_ids"][target.ARMS[-1]], ["d1", "d2"] * 4)
        self.assertEqual(rows[0]["no_rag"], "C")

    def test_paired_stats_count_benefit_and_harm(self):
        stats = target.paired_stats([True, False, True, False], [False, True, True, False])
        self.assertEqual((stats["correct"], stats["wrong_to_correct"],
                          stats["correct_to_wrong"], stats["delta_vs_no_rag"]), (2, 1, 1, 0.))


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import torch

from medrag.generation.pced_pass import decode, pass_token_ids, select_token

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pass_test_official", ROOT / "cache/pced_official/src/methods/pced.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


class FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=[5])

    def __init__(self, steps):
        self.steps, self.inputs, self.masks = steps, [], []

    def __call__(self, input_ids, attention_mask, **kwargs):
        index = len(self.inputs)
        self.inputs.append(input_ids.clone())
        self.masks.append(attention_mask.clone())
        return SimpleNamespace(logits=self.steps[index].unsqueeze(1))


class PassTests(unittest.TestCase):
    def test_exclude_by_raw_not_contrastive_top(self):
        raw = torch.tensor([[0., 1., 9., 0.], [0., 4., 0., 1.]])
        score = torch.tensor([[20., 0., 0., 0.], [0., 4., 0., 1.]])
        token, winner, trace = select_token(raw, score, (2,))
        self.assertEqual((token, winner), (1, 1))
        self.assertEqual(trace["excluded_experts"], [0])

    def test_soft_penalty_changes_winner_without_hard_rejection(self):
        raw = torch.tensor([[2., 0., 1.9], [0., 2., -10.]])
        scores = torch.tensor([[3., 0., 1.9], [0., 2.8, -10.]])
        token, winner, trace = select_token(raw, scores, (2,), penalty=1)
        self.assertEqual(trace["excluded_experts"], [])
        self.assertEqual(trace["ungated_winner_same_prefix"], 0)
        self.assertEqual((token, winner), (1, 1))

    def test_all_pass_fallback_and_no_pass_emission(self):
        raw = torch.tensor([[1., 0., 8.], [0., 4., 6.]])
        token, _, trace = select_token(raw, raw, (2,))
        self.assertNotEqual(token, 2)
        self.assertTrue(trace["all_pass_fallback"])
        self.assertEqual(trace["excluded_experts"], [])

    def test_two_pass_spellings_sum_mass_but_hard_test_is_individual_argmax(self):
        raw = torch.tensor([[2., 1.9, 1.9, 0.]])
        token, _, trace = select_token(raw, raw, (1, 2))
        self.assertGreater(trace["pass_probability"][0], .5)
        self.assertEqual(trace["abstaining_experts"], [])
        self.assertEqual(token, 0)

    def test_prompt_only_never_gates_or_emits_pass(self):
        raw = torch.tensor([[1., 0., 9.], [0., 4., 0.]])
        scores = torch.tensor([[8., 0., 10.], [0., 4., 0.]])
        token, winner, trace = select_token(raw, scores, (2,), enabled=False)
        self.assertEqual((token, winner), (0, 0))
        self.assertFalse(trace["all_pass_fallback"])
        self.assertEqual(trace["excluded_experts"], [])

    def test_nan_is_not_silently_written(self):
        raw = torch.tensor([[float("nan"), 0.]])
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            select_token(raw, raw, (1,))

    def test_real_local_tokenizer(self):
        from transformers import AutoTokenizer
        path = ROOT.parent / "models/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.assertEqual(pass_token_ids(tokenizer), (50892, 51935))
        self.assertGreater(len(tokenizer.encode("<PASS>", add_special_tokens=False)), 1)

    def context(self):
        context = {"input_ids": torch.ones((3, 2), dtype=torch.long),
                   "attention_mask": torch.ones((3, 2), dtype=torch.long)}
        question = {"input_ids": torch.ones((3, 1), dtype=torch.long),
                    "attention_mask": torch.ones((3, 1), dtype=torch.long)}
        return context, question

    def test_rejoin_shared_prefix_one_forward_per_step_first_beta_only(self):
        steps = [torch.tensor([[0., 0., 0., 0., 0., 0.],
                               [0., 0., 9., 0., 0., 0.],
                               [0., 8., 0., 0., 0., 0.]]),
                 torch.tensor([[0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 8., 0., 0.],
                               [0., 0., 9., 0., 0., 0.]]),
                 torch.tensor([[0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 8.],
                               [0., 0., 0., 0., 0., 7.]])]
        model, counter = FakeModel(steps), []
        def jsd(a, b):
            counter.append(1)
            return torch.zeros_like(b)
        c, q = self.context()
        ids, trace = decode(model, None, None, c, q, torch.ones(2), (2,), jsd)
        self.assertEqual(ids, [1, 3, 5])
        self.assertEqual([t["winner"] for t in trace[:2]], [1, 0])
        self.assertEqual(len(counter), 1)
        self.assertEqual(len(model.inputs), 3)
        self.assertTrue(torch.equal(model.inputs[1], torch.full((3, 1), 1)))
        self.assertTrue(torch.equal(model.inputs[2], torch.full((3, 1), 3)))
        self.assertEqual([m.shape[1] for m in model.masks], [3, 4, 5])

    def test_official_noop_parity_with_inactive_pass_ids(self):
        torch.manual_seed(42)
        steps = [torch.randn(3, 8) for _ in range(3)]
        for step in steps:
            step[:, 6:] = -40  # PASS candidates cannot win this synthetic sequence
            step[:, 5] = -30  # no early EOS
        c, q = self.context()
        prior = torch.tensor([.3, .8])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old_ids, old_winners = official.decode_from_cache(FakeModel(steps),
                SimpleNamespace(pad_token_id=5), None, c, q, max_new_tokens=3,
                beta=None, retrieval_scores=prior, gamma=2.5, return_winner_trace=True)
        ids, trace = decode(FakeModel(steps), None, None, c, q, prior, (6, 7), official.get_jsd,
                            enabled=False, max_new_tokens=3)
        self.assertEqual(ids, old_ids.tolist())
        self.assertEqual([t["winner"] for t in trace], old_winners)


if __name__ == "__main__":
    unittest.main()

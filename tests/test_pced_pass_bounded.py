from types import SimpleNamespace
import unittest

import torch

from medrag.generation.pced_pass_bounded import Stream, generate, select


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[5])

    def __init__(self, outputs):
        self.outputs, self.inputs, self.mask_lengths = outputs, [], []

    def __call__(self, input_ids, attention_mask, **kwargs):
        result = self.outputs[len(self.inputs)]
        self.inputs.append(input_ids.clone())
        self.mask_lengths.append(attention_mask.shape[1])
        return SimpleNamespace(logits=result[:, None, :])


class BoundedPassTests(unittest.TestCase):
    def test_documented_candidate_specific_penalty_changes_winner(self):
        raw = torch.tensor([[.3, .5, .2], [.19, .8, .01]]).log()
        base = torch.tensor([[10., -5., -5.], [9.4, -5., -5.]])
        amateur = torch.zeros(3)
        self.assertEqual(select(raw, amateur, base, (2,), "exclude")[1]["winner"], 0)
        token, trace = select(raw, amateur, base, (2,), "bounded", penalty=4)
        self.assertEqual((token, trace["winner"]), (0, 1))
        self.assertAlmostEqual(trace["bounded_best_scores"][0], 8.4, places=5)
        self.assertAlmostEqual(trace["bounded_best_scores"][1], 9.2, places=5)
        self.assertEqual(trace["excluded_experts"], [])

    def test_penalty_can_change_token_within_same_expert(self):
        raw = torch.tensor([[4., 1., 2.]])
        base = torch.tensor([[5., 6., 0.]])
        self.assertEqual(select(raw, raw[0], base, (2,), "exclude")[0], 1)
        self.assertEqual(select(raw, raw[0], base, (2,), "bounded")[0], 0)

    def test_tiny_pass_has_no_unbounded_bonus_or_probability_floor(self):
        raw = torch.tensor([[2., 1., -100.], [2., 1., -1000.]])
        base = torch.tensor([[5., 0., 0.], [4., 0., 0.]])
        _, trace = select(raw, raw[0], base, (2,), "bounded")
        self.assertEqual(trace["winner"], 0)
        self.assertEqual(trace["bounded_best_scores"], [5., 4.])
        shifted = raw+torch.tensor([[100.], [-200.]])
        _, other = select(shifted, raw[0], base, (2,), "bounded")
        self.assertEqual(trace["bounded_best_scores"], other["bounded_best_scores"])

    def test_hard_exclusion_and_all_pass_amateur_fallback(self):
        raw = torch.tensor([[0., 1., 9.], [3., 0., 9.]])
        amateur = torch.tensor([1., 7., 20.])
        for arm in ("exclude", "bounded"):
            token, trace = select(raw, amateur, raw, (2,), arm)
            self.assertEqual(token, 1)
            self.assertEqual(trace["winner"], -1)
            self.assertEqual(trace["excluded_experts"], [0, 1])
            self.assertTrue(trace["amateur_fallback"])
        self.assertFalse(select(raw, amateur, raw, (2,), "prompt_only")[1]["amateur_fallback"])

    def test_pass_alias_mass_and_raw_top1_remain_distinct(self):
        raw = torch.tensor([[2., 1.9, 1.9, 0.]])
        _, trace = select(raw, raw[0], raw, (1, 2), "bounded")
        self.assertGreater(trace["pass_probability"][0], .5)
        self.assertGreater(trace["selected_token_pass_ratio"][0], .5)
        self.assertEqual(trace["excluded_experts"], [])

    def test_forbidden_eos_cannot_win_amateur_fallback(self):
        raw = torch.tensor([[0., 2., 9., 20.]])
        token, trace = select(raw, torch.tensor([1., 8., 9., 30.]), raw,
                              (2,), "bounded", forbidden=(3,))
        self.assertEqual(token, 1)
        self.assertEqual(trace["raw_top_token_ids"], [3])
        self.assertEqual(trace["format_top_token_ids"], [2])
        self.assertTrue(trace["amateur_fallback"])

    def test_pass_reentry_and_shared_tokens_use_one_forward(self):
        outputs = [torch.tensor([[0.]*6, [0., 0., 9., 0., 0., 0.], [0., 8., 0., 0., 0., 0.]]),
                   torch.tensor([[0.]*6, [0., 0., 0., 8., 0., 0.], [0., 0., 9., 0., 0., 0.]])]
        model = FakeModel(outputs)
        context = {"attention_mask": torch.ones(3, 2, dtype=torch.long)}
        suffix = {"input_ids": torch.ones(3, 1, dtype=torch.long), "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        jsd_calls = []
        def jsd(a, b):
            jsd_calls.append(1)
            return torch.zeros_like(b)
        stream = Stream(model, None, context, suffix, torch.ones(2), (2,), jsd, "bounded")
        self.assertEqual(stream.step("reasoning", 0)[0], 1)
        stream.force([4, 4])
        self.assertEqual(stream.step("reasoning", 1)[0], 3)
        self.assertEqual([t["winner"] for t in stream.traces], [1, 0])
        self.assertEqual(model.inputs[1].tolist(), [[1, 4, 4]]*3)
        self.assertEqual(model.mask_lengths, [3, 6])
        self.assertEqual(len(jsd_calls), 1)

    def test_leading_newline_and_split_answer_marker(self):
        class Tokens:
            def decode(self, ids, **kwargs):
                return "".join({0: " \n", 1: "A fact.\n", 2: "Ans", 3: "wer:", 4: " Paris", 5: ""}[i] for i in ids)
            def encode(self, text, **kwargs): return [8]
        class Scripted:
            def __init__(self):
                self.values = iter([(0, False), (1, False), (2, False), (3, False), (4, False), (5, True)])
                self.traces, self.permissions, self.controls = [], [], []
            def step(self, phase, local, *, allow_eos):
                self.permissions.append(allow_eos)
                return next(self.values)
            def force(self, ids): self.controls.append(ids)
        stream = Scripted()
        result = generate(stream, Tokens())
        self.assertEqual(result["reasoning"], "A fact.")
        self.assertEqual(result["prediction"], "Paris")
        self.assertEqual(result["reasoning_stop"], "answer_marker")
        self.assertEqual(stream.permissions, [False]*5+[True])
        self.assertEqual(stream.controls, [])

    def test_budget_forces_header_and_empty_answer_does_not_unlock_eos(self):
        class Tokens:
            def decode(self, ids, **kwargs): return "".join({0: " ", 1: "unfinished"}[i] for i in ids)
            def encode(self, text, **kwargs): return [8, 9]
        class Scripted:
            def __init__(self): self.traces, self.controls, self.permissions = [], [], []
            def step(self, phase, local, *, allow_eos):
                self.permissions.append(allow_eos)
                return (1 if phase == "reasoning" else 0), False
            def force(self, ids): self.controls.append(ids)
        stream = Scripted()
        result = generate(stream, Tokens(), reasoning_tokens=2, answer_tokens=2)
        self.assertEqual(stream.controls, [[8, 9]])
        self.assertEqual(stream.permissions, [False]*4)
        self.assertTrue(result["empty_answer"])
        self.assertEqual(result["reasoning_stop"], "token_budget")

    def test_real_cpu_llama_cached_forward_and_forced_header(self):
        # Tiny untrained fixture validates HF mask/cache execution, not quality.
        from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128, eos_token_id=31)).eval()
        model.requires_grad_(False)
        context = {"input_ids": torch.ones(3, 2, dtype=torch.long), "attention_mask": torch.ones(3, 2, dtype=torch.long)}
        suffix = {"input_ids": torch.ones(3, 1, dtype=torch.long), "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        cache = DynamicCache()
        with torch.inference_mode():
            model.model(**context, past_key_values=cache, use_cache=True)
        stream = Stream(model, cache, context, suffix, torch.ones(2), (29, 30),
                        lambda a, b: torch.zeros_like(b), "bounded")
        self.assertFalse(stream.step("reasoning", 0)[1])
        stream.force([7, 8])
        stream.step("reasoning", 1)
        self.assertEqual(stream.cache.get_seq_length(), 6)


if __name__ == "__main__":
    unittest.main()

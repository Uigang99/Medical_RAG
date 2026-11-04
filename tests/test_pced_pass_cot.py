from types import SimpleNamespace
import unittest

import torch

from medrag.generation.pced_pass_cot import ARMS, Stream, generate, select


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[5])

    def __init__(self, outputs):
        self.outputs, self.inputs, self.mask_lengths = outputs, [], []

    def __call__(self, input_ids, attention_mask, **kwargs):
        output = self.outputs[len(self.inputs)]
        self.inputs.append(input_ids.clone())
        self.mask_lengths.append(attention_mask.shape[1])
        return SimpleNamespace(logits=output[:, None, :])


class PassCotTests(unittest.TestCase):
    def test_ratio_changes_winner_without_hard_block(self):
        raw = torch.tensor([[.6, .3, .1], [.5, .499, .001]]).log()
        self.assertEqual(select(raw, raw, (2,), "probability")[1], 0)
        token, winner, trace, _ = select(raw, raw, (2,), "pass_dynamic")
        self.assertEqual((token, winner), (0, 1))
        self.assertEqual(trace["excluded_experts"], [])

    def test_floor_prevents_rewarding_arbitrarily_tiny_pass(self):
        raw = torch.tensor([[.6, .4, 1e-8], [.5, .5, 1e-20]]).log()
        self.assertEqual(select(raw, raw, (2,), "pass_dynamic", floor=1e-4)[1], 0)

    def test_static_freezes_both_exclusion_and_soft_mass(self):
        first = torch.tensor([[0., 0., 8.], [8., 0., 0.]])
        _, _, _, frozen = select(first, first, (2,), "pass_static")
        current = torch.tensor([[8., 0., 0.], [0., 0., 8.]])
        self.assertEqual(select(current, current, (2,), "pass_static", frozen=frozen)[1], 1)
        self.assertEqual(select(current, current, (2,), "pass_dynamic", frozen=frozen)[1], 0)

    def test_all_pass_restores_for_one_position_only(self):
        raw = torch.tensor([[0., 2., 8.], [3., 0., 8.]])
        token, _, trace, _ = select(raw, raw, (2,), "pass_dynamic")
        self.assertNotEqual(token, 2)
        self.assertTrue(trace["all_pass_fallback"])
        self.assertEqual(trace["excluded_experts"], [])

    def test_pass_alias_sum_does_not_replace_raw_top1_test(self):
        raw = torch.tensor([[2., 1.9, 1.9, 0.]])
        _, _, t, _ = select(raw, raw, (1, 2), "pass_dynamic")
        self.assertGreater(t["pass_probability"][0], .5)
        self.assertFalse(t["excluded_experts"])

    def test_normalized_score_shift_invariance_and_finite_guard(self):
        raw = torch.tensor([[4., 0., -1.], [0., 3., -4.]])
        for arm in ARMS[1:]:
            a = select(raw, raw, (2,), arm)[:2]
            shifted = raw + torch.tensor([[100.], [-20.]])
            self.assertEqual(a, select(shifted, shifted, (2,), arm)[:2])
        raw[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            select(raw, raw, (2,), "pass_dynamic")

    def test_shared_token_reentry_and_batched_header_cache_lengths(self):
        outputs = [torch.tensor([[0.]*6, [0., 0., 9., 0., 0., 0.], [0., 8., 0., 0., 0., 0.]]),
                   torch.tensor([[0.]*6, [0., 0., 0., 8., 0., 0.], [0., 0., 9., 0., 0., 0.]]),
                   torch.tensor([[0.]*6, [0., 0., 0., 0., 0., 9.], [0., 0., 0., 0., 0., 8.]])]
        model = FakeModel(outputs)
        c = {"attention_mask": torch.ones(3, 2, dtype=torch.long)}
        q = {"input_ids": torch.ones(3, 1, dtype=torch.long), "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        calls = []
        def jsd(a, b):
            calls.append(1)
            return torch.zeros_like(b)
        stream = Stream(model, None, c, q, torch.ones(2), (2,), jsd, "pass_dynamic", 1e-4)
        self.assertEqual(stream.step("reasoning", 1, 0)[0], 1)
        stream.force([4, 4])
        self.assertEqual(stream.step("reasoning", 2, 0)[0], 3)
        self.assertTrue(stream.step("answer", 0, 0)[1])
        self.assertEqual([t["winner"] for t in stream.traces[:2]], [1, 0])
        self.assertEqual(model.mask_lengths, [3, 6, 7])
        self.assertEqual(model.inputs[1].tolist(), [[1, 4, 4]]*3)
        self.assertEqual(len(calls), 1)

    def test_protocol_early_finish_and_answer_only(self):
        class Tokens:
            def encode(self, text, **kwargs): return [9]
            def decode(self, ids, **kwargs): return "".join({1: "A fact.\n", 2: "Correct answer", 5: ""}[i] for i in ids)
        class Scripted:
            traces = []
            def __init__(self): self.values = iter([(1, False), (5, True), (2, False), (5, True)])
            def step(self, *args): return next(self.values)
            def force(self, ids): pass
        result = generate(Scripted(), Tokens())
        self.assertEqual(result["prediction"], "Correct answer")
        self.assertEqual(len(result["reasoning"]), 2)
        self.assertEqual(result["reasoning_stop"], "empty_eos_ready")
        self.assertEqual(result["answer_status"], "eos")

    def test_real_cpu_llama_cache_path(self):
        # Untrained tiny CPU fixture: validates HF cache/mask API, not task quality.
        from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128, eos_token_id=31)).eval()
        model.requires_grad_(False)
        context = {"input_ids": torch.ones(3, 2, dtype=torch.long),
                   "attention_mask": torch.ones(3, 2, dtype=torch.long)}
        suffix = {"input_ids": torch.ones(3, 1, dtype=torch.long),
                  "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        cache = DynamicCache()
        with torch.inference_mode():
            model.model(**context, past_key_values=cache, use_cache=True)
        stream = Stream(model, cache, context, suffix, torch.ones(2), (29, 30),
                        lambda a, b: torch.zeros_like(b), "pass_dynamic", 1e-4)
        stream.step("reasoning", 1, 0)
        stream.force([7, 8])
        stream.step("reasoning", 2, 0)
        self.assertEqual(len(stream.traces), 2)
        self.assertGreaterEqual(stream.cache.get_seq_length(), 5)


if __name__ == "__main__":
    unittest.main()

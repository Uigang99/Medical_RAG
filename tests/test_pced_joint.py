from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import warnings

import torch

from medrag.generation.pced_joint import (JointConfig, joint_prior, probability_jsd,
    select_pair, pced_scores, CachedStream, JointStream, make_joint_builder, generate)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"cache/pced_official"))
from src.methods.pced import get_jsd, prefill, decode_from_cache


class JointTests(unittest.TestCase):
    def test_prior_is_mean_in_log_space_not_mean_raw_score(self):
        self.assertAlmostEqual(joint_prior(.9, .1), .3)
        self.assertAlmostEqual(joint_prior(.9, .1, "arithmetic"), .5)
        self.assertEqual(joint_prior(.9, .1, "minimum"), .1)
        for values in ((-1, .1), (float("nan"), .2), (2., .1)):
            with self.assertRaises(ValueError):
                joint_prior(*values)

    def test_jsd_filters_effect_and_relevance_and_no_repeated_pair(self):
        amateur = torch.zeros(1, 3)
        expert = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
        priors = torch.tensor([1., 1., .01])
        cfg = JointConfig()
        pair, details = select_pair(amateur, expert, priors, cfg)
        self.assertEqual(pair, (0, 1))
        self.assertEqual(details["eligible_experts"], [1, 2])
        self.assertIsNone(select_pair(amateur, expert, priors, cfg, used=[(0, 1)])[0])
        self.assertIsNone(select_pair(amateur, torch.zeros_like(expert), priors, cfg)[0])
        shifted = expert + torch.tensor([[100.], [-70.], [21.]])
        self.assertAlmostEqual(details["pair_jsd"], select_pair(amateur+10, shifted, priors, cfg)[1]["pair_jsd"], places=5)

    def test_identical_delta_distributions_do_not_activate(self):
        cfg = JointConfig()
        pair, _ = select_pair(torch.zeros(1, 3), torch.tensor([[8., 0., 0.], [9., 1., 1.]]), torch.ones(2), cfg)
        self.assertIsNone(pair)
        self.assertEqual(float(probability_jsd(torch.tensor([.5, .5]), torch.tensor([.5, .5]))), 0)

    def test_maximum_over_eligible_pairs_not_just_highest_priors(self):
        experts = torch.tensor([[6., 0., 0.], [5., 0., 0.], [0., 8., 0.]])
        pair, info = select_pair(torch.zeros(1, 3), experts, torch.tensor([1., .9, .8]), JointConfig())
        self.assertEqual(pair, (0, 2))
        self.assertEqual(info["eligible_pairs"], 3)

    def test_joint_token_broadcast_and_lifetime(self):
        class Model:
            generation_config = SimpleNamespace(eos_token_id=9)
            def __init__(self): self.calls = []
            def __call__(self, input_ids, attention_mask, **kw):
                self.calls.append(input_ids.tolist())
                logits = torch.zeros(input_ids.shape[0], 1, 10)
                logits[:, :, 1 if input_ids.shape[0] == 3 else 2] = 9
                if input_ids.shape[0] == 1: logits[:, :, 2] = 20
                return SimpleNamespace(logits=logits, past_key_values=None)
        model = Model()
        c = {"attention_mask": torch.ones(3, 1, dtype=torch.long)}
        s = {"input_ids": torch.ones(3, 1, dtype=torch.long), "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        histories = []
        def builder(pair, history):
            histories.append(history)
            ids = torch.tensor([[4, 1]+history])
            return CachedStream(model, None, ids, torch.ones_like(ids)), {}
        cfg = JointConfig(warmup=0, check_every=1, lifetime=2, max_creations=1, min_effect_jsd=0, min_pair_jsd=0)
        stream = JointStream(model, None, c, s, torch.ones(2), get_jsd, config=cfg, joint_builder=builder)
        self.assertEqual(stream.step("reasoning")[0], 2)
        stream.force([7, 8])
        self.assertEqual(stream.step("reasoning")[0], 2)
        self.assertEqual(model.calls[2], [[2, 7, 8]]*3)
        self.assertEqual(model.calls[3], [[2, 7, 8]])
        self.assertIsNone(stream.joint)
        self.assertEqual(stream.events[0]["end_reason"], "lifetime")
        self.assertEqual(stream.step("answer")[0], 1)
        self.assertEqual(stream.history, [2, 7, 8, 2, 1])
        self.assertEqual(histories, [[]])

    def tiny_model(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(123)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128, eos_token_id=31)).eval()
        model.requires_grad_(False)
        return model

    def test_disabled_joint_matches_pinned_official_decoder_cpu_llama(self):
        model = self.tiny_model()
        context = {"input_ids": torch.tensor([[0, 1, 2], [3, 4, 5], [0, 6, 7]]),
                   "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 1]])}
        suffix = {"input_ids": torch.tensor([[8, 9]]*3), "attention_mask": torch.ones(3, 2, dtype=torch.long)}
        prior = torch.tensor([.8, .7])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected, winners = decode_from_cache(model, SimpleNamespace(pad_token_id=0), prefill(model, context),
                context, suffix, max_new_tokens=6, beta=None, retrieval_scores=prior, gamma=2.5, return_winner_trace=True)
        stream = JointStream(model, prefill(model, context), context, suffix, prior, get_jsd)
        observed = []
        for _ in range(6):
            token, eos = stream.step("reasoning")
            observed.append(token)
            if eos: break
        self.assertEqual(observed, expected.tolist())
        self.assertEqual([x["winner_rank"]-1 for x in stream.trace], winners)

    def test_joint_builder_uses_both_documents_and_exact_shared_history(self):
        model = self.tiny_model()
        class Tokenizer:
            pad_token_id = 0
            def encode(self, text, **kwargs):
                self.seen = text
                return [3, 4, 5]
        tokenizer = Tokenizer()
        row = {"question": "Q", "texts": ["doc A", "doc B"], "suffix_ids": [8, 9], "scores": {"pids": ["a", "b"]}}
        def prefixes(tok, q, texts): return ["amateur", texts[0]], "unused"
        builder = make_joint_builder(model, tokenizer, row, prefixes, 5)
        history = [11, 12]
        stream, meta = builder((0, 1), history)
        self.assertEqual(tokenizer.seen, "doc A\n\ndoc B")
        self.assertEqual(meta["left_padding"], 2)
        full = stream.pending.clone()
        mask = stream.mask.clone()
        first = stream.logits()
        with torch.inference_mode():
            reference = model(input_ids=full, attention_mask=mask, use_cache=False).logits[:, -1]
        torch.testing.assert_close(first, reference)
        stream.queue([13, 7, 8])  # shared generated token + forced control tokens
        actual = stream.logits()
        full = torch.cat((full, torch.tensor([[13, 7, 8]])), -1)
        mask = torch.cat((mask, torch.ones(1, 3, dtype=torch.long)), -1)
        with torch.inference_mode():
            reference = model(input_ids=full, attention_mask=mask, use_cache=False).logits[:, -1]
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            builder((0, 1), [1]*128)

    def test_eos_not_inserted_and_no_reasoning_credit(self):
        class Tokens:
            def encode(self, text, **kwargs): return [7, 8]
            def decode(self, ids, **kwargs): return "gold" if ids == [1] else "wrong" if ids == [2] else ""
        class Model:
            generation_config = SimpleNamespace(eos_token_id=9)
            def __init__(self): self.calls = []; self.tokens = iter([1, 9, 2, 9])
            def __call__(self, input_ids, attention_mask, **kw):
                self.calls.append(input_ids.tolist())
                logits = torch.zeros(2, 1, 10)
                logits[:, :, next(self.tokens)] = 10
                return SimpleNamespace(logits=logits, past_key_values=None)
        model = Model()
        context = {"attention_mask": torch.ones(2, 1, dtype=torch.long)}
        suffix = {"input_ids": torch.ones(2, 1, dtype=torch.long), "attention_mask": torch.ones(2, 1, dtype=torch.long)}
        stream = JointStream(model, None, context, suffix, torch.ones(1), get_jsd)
        result = generate(stream, Tokens(), reasoning_tokens=5, answer_tokens=5)
        self.assertEqual(result["reasoning_text"], "gold")
        self.assertEqual(result["prediction"], "wrong")
        self.assertEqual(result["shared_history_ids"], [1, 7, 8, 2])
        self.assertEqual(model.calls[2], [[7, 8]]*2)


if __name__ == "__main__":
    unittest.main()

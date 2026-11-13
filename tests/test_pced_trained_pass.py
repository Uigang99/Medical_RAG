"""Tests of the actual gating invariant, state/cache handling, and audit records."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from medrag.generation.pced_trained_pass import (choose, pass_probabilities, initial_beta,
                                               trace_frame, save_trace, Engine)


class Decisions(unittest.TestCase):
    def setUp(self):
        self.z = torch.tensor([[1., 6., 0., -8.], [7., 0., 5., -8.], [1., 2., 8., -8.]])
        self.prior = torch.tensor([.6, .4]).log()

    def select(self, probs, **kwargs):
        p = torch.tensor(probs)
        return choose(self.z, .2, self.prior, p.log(), (1-p).log(), gated=True,
                      threshold=.7, forbidden=(3,), **kwargs)

    def test_no_active_fallbacks(self):
        d = self.select([.9, .9])
        self.assertEqual(d["mode"], 2)
        self.assertEqual(d["token"], d["counter_token"])
        amateur = self.select([.9, .9], fallback="amateur")
        self.assertEqual((amateur["winner"], amateur["token"]), (0, 1))

    def test_single_expert_uses_raw_and_reentry(self):
        d = self.select([.1, .9])
        self.assertEqual((d["winner"], d["token"], d["mode"]), (1, 0, 3))
        other = self.select([.9, .1])
        self.assertEqual((other["winner"], other["token"]), (2, 2))

    def test_multi_expert_penalty_can_change_winner(self):
        z = torch.tensor([[0., 0., -9.], [5., 4., -9.], [4., 5.1, -9.]])
        p = torch.tensor([.01, .69])
        d = choose(z, .0, torch.zeros(2), p.log(), (1-p).log(), gated=True, threshold=.7,
                   penalty=1., forbidden=(2,))
        self.assertEqual((d["counter_winner"], d["counter_token"]), (2, 1))
        self.assertEqual((d["winner"], d["token"]), (1, 0))

    def test_train_probability_denominator_and_no_pass_emission(self):
        h = torch.randn(3, 5)
        w, b, logits = torch.randn(5), torch.tensor(.2), torch.randn(3, 8)
        zp, lp, ln = pass_probabilities(h, logits, w, b)
        reference = torch.cat((logits, zp[:, None]), -1).log_softmax(-1)
        torch.testing.assert_close(lp, reference[:, -1])
        torch.testing.assert_close(ln.exp() + lp.exp(), torch.ones(3))
        self.z[:, 3] = 1000
        self.assertNotEqual(self.select([.2, .3])["token"], 3)

    def test_trace_has_every_candidate_and_exact_components(self):
        d = self.select([.2, .3])
        h = torch.randn(3, 5, dtype=torch.bfloat16)
        frame = trace_frame(h, self.z, d, log_p=torch.tensor([.2, .3]).log(),
            log_not=torch.tensor([.8, .7]).log(), pass_logit=torch.zeros(2), top_n=2, forbidden=(3,))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.npz"
            save_trace(path, [frame, frame], beta=.2, log_prior=self.prior.numpy())
            with np.load(path) as data:
                hidden = torch.from_numpy(data["hidden_bf16"][0].copy()).view(torch.bfloat16)
                torch.testing.assert_close(hidden, h, rtol=0, atol=0)
                start, end = data["candidate_offsets"][:2]
                np.testing.assert_array_equal(data["candidate_logits"][start:end].T,
                                              self.z[:, data["candidate_ids"][start:end]].numpy())


class TinyRealCache(unittest.TestCase):
    """A tiny Llama exercises the same PEFT layers and HF causal cache on CPU."""
    @classmethod
    def setUpClass(cls):
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        from types import SimpleNamespace
        torch.manual_seed(2)
        config = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, eos_token_id=39)
        base = LlamaForCausalLM(config).eval()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.data.normal_(std=.01)
        cls.engine = Engine(model.eval(), SimpleNamespace(pad_token_id=0, eos_token_id=39),
                            {"w": torch.zeros(32), "b": torch.tensor(0.), "pass_id": 38})

    def test_batch_single_and_frozen_amateur(self):
        groups = [[[2, 4], [2, 4, 5], [2, 4, 6, 8]], [[2, 7, 3], [2, 7], [2, 7, 8]]]
        prior = [[-.3, -.4], [-.1, -.8]]
        batch = self.engine.competitive(groups, prior, "pass_pced", 4)
        for i in range(2):
            solo = self.engine.competitive(groups[i:i+1], prior[i:i+1], "pass_pced", 4)
            self.assertEqual(batch[i].ids, solo[0].ids)
        baseline = self.engine.competitive(groups, prior, "pced", 1)
        adapted = self.engine.competitive(groups, prior, "pced_lora", 1)
        np.testing.assert_allclose(baseline[0].frames[0]["raw_top_logits"][0],
                                   adapted[0].frames[0]["raw_top_logits"][0], atol=1e-6)

    def test_masked_experts_consume_prefix(self):
        from medrag.generation.pced_trained_pass import StreamBatch
        prompts = [[[2, 3], [2, 3, 4], [2, 3, 5]]]
        gen = self.engine.competitive(prompts, [[-.2, -.4]], "pass_pced", 3, threshold=0.)[0]
        self.assertEqual([int(f["mode"]) for f in gen.frames], [2] * len(gen.frames))
        with torch.inference_mode():
            full = StreamBatch(self.engine, [p+gen.ids[:2] for p in prompts[0][1:]], True)
            frame = gen.frames[2]
            np.testing.assert_allclose(full.logits[:, frame["candidate_ids"]].cpu().numpy().T,
                                       frame["candidate_logits"][:, 1:], atol=1e-5)

    def test_end_to_end_records_reader_resume_and_summary(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import evaluate_pass_lora_qa800 as run
        tok = SimpleNamespace(encode=lambda text, **kw: [2, 5],
                              decode=lambda ids, **kw: "blue",
                              apply_chat_template=lambda *args, **kw: "prompt")
        progress = SimpleNamespace(render=lambda *args: None)
        with tempfile.TemporaryDirectory() as folder:
            a = run.arguments(["--output-root", folder, "--per-dataset", "1",
                               "--reasoning-tokens", "8", "--answer-tokens", "8"])
            rows = [dict(dataset=ds, qid=ds+"1", question="What color?", answers=["blue"],
                         pids=["d1", "d2"], log_prior=[-.2, -.4],
                         tokens={"no_rag": [2, 3], "all": [2, 3, 4, 5],
                                 "experts": [[2, 3], [2, 3, 4], [2, 3, 5]]}) for ds in run.DATASETS]
            with patch.object(run, "resource_check", return_value={}):
                for arm in a.arms:
                    run.process_batch(a, rows, arm, self.engine, tok, "test-contract", progress, 0.)
                # Existing reasoning must survive an interrupted/repeated reader phase.
                with patch.object(self.engine, "competitive", side_effect=AssertionError("regenerated reasoning")):
                    run.process_batch(a, rows[:1], "pass_pced", self.engine, tok, "test-contract", progress, 0.)
            report = run.summarize(a, rows, "test-contract")
            self.assertEqual(report["conditions"]["pass_pced"]["groups"]["ALL"]["correct"], 8)
            self.assertEqual(report["paired"]["pass_pced_minus_pced_lora"]["ALL"]["delta_pp"], 0)
            with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                run.cached(a, rows[0], "pass_pced", "different-contract")


if __name__ == "__main__":
    unittest.main()

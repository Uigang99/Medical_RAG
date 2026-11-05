from types import SimpleNamespace
import unittest

import torch
from transformers import DynamicCache

from medrag.generation import pced_sentence_pass as m
from medrag.generation import pced_sentence_reuse as fast


class FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=[11])
    def __init__(self): self.calls, self.same_token = 0, False
    def prefill(self, ids, cache):
        k = ids.float()[:, None, :, None]
        cache.update(k, k.clone(), 0)
    def __call__(self, input_ids, past_key_values, **kwargs):
        self.calls += 1
        self.prefill(input_ids, past_key_values)
        z = torch.zeros((3, 1, 12))
        # Each expert proposes a different token. The common history affects
        # logits, so a cache contaminated by another rollout cannot pass parity.
        history = past_key_values.layers[0].keys[:, 0, :, 0].sum(-1)%3
        z[0, 0, 1] = 4
        z[1, 0, 2] = 5+history[1]*.01
        z[2, 0, 2 if self.same_token else 3] = 5.2+history[2]*.01
        return SimpleNamespace(logits=z, past_key_values=past_key_values)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.context = {"input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                        "attention_mask": torch.ones(3, 2, dtype=torch.long)}
        self.suffix = {"input_ids": torch.ones(3, 1, dtype=torch.long), "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        self.prior = torch.tensor([.6, .4])
        self.jsd = lambda a, b: torch.zeros_like(b)

    def prefill(self):
        cache = DynamicCache()
        self.model.prefill(self.context["input_ids"], cache)
        return cache

    def run_stream(self, arm, judge, anchor=None):
        cls = fast.ReplayStream if anchor else m.Stream
        stream = cls(self.model, None if anchor else self.prefill(), self.context, self.suffix,
                     self.prior, (10,), self.jsd, arm, None if arm == "pced" else judge,
                     **({"anchor": anchor} if anchor else {}))
        ids = []
        for i in range(8):
            if i in (0, 3, 6):
                stream.begin_segment(str(ids), "reasoning", "boundary")
            ids.append(stream.step("reasoning", i, allow_eos=False)[0])
        return stream, ids

    def test_replay_and_partial_fork_equal_independent_rollouts(self):
        def judge(text, phase):
            return {"contribute_probability": [1., 1.] if text == "[]" else [.51, .99], "seconds": 1.}
        memo = fast.MemoJudge(judge)
        anchor, anchor_ids = self.run_stream("sentence_weighted", memo)
        anchor_k = anchor.cache.layers[0].keys.clone()
        for arm in ("pced", "sentence_exclude"):
            replay, ids = self.run_stream(arm, memo, anchor)
            standalone, expected = self.run_stream(arm, judge)
            self.assertEqual(ids, expected)
            self.assertEqual(replay.reused_steps, 3)
            self.assertNotEqual(ids, anchor_ids)
            for x, y in zip(replay.traces, standalone.traces):
                for key in ("token_id", "winner", "active_experts", "excluded_experts", "weighted_best_scores", "counterfactual_same_prefix", "selected_token_raw_logits"):
                    self.assertEqual(x[key], y[key], key)
            torch.testing.assert_close(anchor.cache.layers[0].keys, anchor_k, atol=0, rtol=0)
        self.assertGreater(memo.hits, 0)

    def test_full_replay_uses_no_forward_and_changes_winner_when_token_equal(self):
        judge = lambda text, phase: {"contribute_probability": [1., 1.], "seconds": 1.}
        anchor, ids = self.run_stream("sentence_weighted", judge)
        before = self.model.calls
        replay, other = self.run_stream("pced", judge, anchor)
        self.assertEqual(other, ids)
        self.assertEqual(replay.reused_steps, 8)
        self.assertIsNone(replay.cache)
        self.assertEqual(self.model.calls, before)

    def test_fork_and_memo_do_not_alias_mutable_data(self):
        cache = self.prefill()
        fork = fast.fork_prefix(cache, 1)
        fork.layers[0].keys.zero_()
        self.assertTrue(bool(cache.layers[0].keys.any()))
        with self.assertRaises(ValueError): fast.fork_prefix(cache, 3)
        memo = fast.MemoJudge(lambda text, phase: {"contribute_probability": [.7, .8], "seconds": 2.})
        first = memo("A", "reasoning")
        first["contribute_probability"][0] = 0
        second = memo("A", "reasoning")
        self.assertEqual(second["contribute_probability"][0], .7)
        self.assertEqual(second["seconds"], 0)
        self.assertEqual((memo.hits, memo.misses), (1, 1))
        memo("A", "answer")
        self.assertEqual(memo.misses, 2)

    def test_same_token_different_winner_still_reuses_exact_trajectory(self):
        self.model.same_token = True
        judge = lambda text, phase: {"contribute_probability": [.51, .99], "seconds": 0.}
        anchor, ids = self.run_stream("sentence_weighted", judge)
        before = self.model.calls
        replay, other = self.run_stream("pced", judge, anchor)
        self.assertEqual(ids, other)
        self.assertEqual(self.model.calls, before)
        self.assertEqual(replay.reused_steps, 8)
        self.assertNotEqual(anchor.traces[0]["winner"], replay.traces[0]["winner"])
        standalone, expected = self.run_stream("pced", judge)
        for x, y in zip(replay.traces, standalone.traces):
            self.assertEqual(x["counterfactual_same_prefix"], y["counterfactual_same_prefix"])

    def test_all_pass_fallback_replay_and_reentry_are_identical(self):
        def judge(text, phase):
            count = text.count(",")+1 if text != "[]" else 0
            return {"contribute_probability": [0., 0.] if 3 <= count < 6 else [1., 1.], "seconds": 0.}
        anchor, ids = self.run_stream("sentence_weighted", judge)
        for arm in ("sentence_exclude", "pced"):
            replay, other = self.run_stream(arm, judge, anchor)
            standalone, expected = self.run_stream(arm, judge)
            self.assertEqual(other, expected)
            for x, y in zip(replay.traces, standalone.traces):
                for key in ("winner", "all_pass", "amateur_fallback", "excluded_experts"):
                    self.assertEqual(x[key], y[key])

    def test_real_cpu_llama_cache_fork_runtime_check(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128, eos_token_id=15)).eval()
        def prefill(model, ctx):
            cache = DynamicCache()
            model.model(**ctx, past_key_values=cache, use_cache=True)
            return cache
        result = fast.check_live_fork(model, self.context, self.suffix, self.prior, (14,), self.jsd, prefill)
        self.assertTrue(result["passed"])

    def test_full_arguments_keep_current_pass_contract(self):
        from scripts.evaluate_pced_sentence_pass_800 import arguments, DATASETS
        a = arguments([])
        self.assertEqual(len(DATASETS), 8)
        self.assertEqual((a.reasoning_tokens, a.answer_tokens, a.contribution_weight), (192, 32, 4.))
        self.assertEqual(a.judge_batch_size, 16)
        self.assertTrue(a.reuse_arm_prefixes)
        self.assertFalse(arguments(["--independent-arms"]).reuse_arm_prefixes)

    def test_streaming_summary_counts_incomplete_as_wrong_and_validates_resume(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scripts import evaluate_pced_sentence_pass_800 as run
        with TemporaryDirectory() as tmp:
            a = SimpleNamespace(output_root=Path(tmp), arms=list(m.ARMS))
            rows = [{"dataset": ds, "qid": "q", "answers": ["Gold"]} for ds in run.DATASETS]
            for arm in a.arms:
                for i, row in enumerate(rows):
                    incomplete = arm == "sentence_weighted" and i == 0
                    r = dict(row, contract_hash="test", arm=arm, subspan_em=int(not incomplete),
                        prediction="" if incomplete else "Gold", empty_answer=incomplete, empty_reasoning=False,
                        reasoning_stop="token_budget" if incomplete else "answer_marker",
                        answer_status="skipped_incomplete_reasoning" if incomplete else "eos",
                        reasoning_token_ids=[1, 2], trace=[], decisions=[],
                        reused_generation_positions=0, seconds=1., judge_seconds=0.)
                    run.paper.atomic_json(run.common.output_path(a, arm, row), r)
            result = run.summarize(a, rows, "test")
            self.assertEqual(result["conditions"]["sentence_weighted"]["n"], 8)
            self.assertEqual(result["conditions"]["sentence_weighted"]["correct"], 7)
            self.assertEqual(result["conditions"]["sentence_weighted"]["macro8"], 7/8)
            self.assertEqual(result["conditions"]["sentence_weighted"]["answer_skipped_incomplete_reasoning"], 1)
            self.assertEqual(result["paired"]["sentence_weighted_vs_pced"]["ALL"]["c_to_w"], 1)
            with self.assertRaises(RuntimeError):
                run.cached(a, "pced", rows[0], "wrong-contract")


if __name__ == "__main__": unittest.main()

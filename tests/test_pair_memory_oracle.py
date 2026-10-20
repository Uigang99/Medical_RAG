import unittest
from pathlib import Path
import tempfile
import copy

import torch

from medrag.generation.pair_memory_attention import (
    PairMemoryBridge, FrozenLlamaMemory, apply_messages, answer_loss)
from scripts.pilot_rag2_pair_memory_oracle import (stats, better, Store, oracle,
    baseline_audit, ensure_resume_contract, RUNNER, INITIAL_RUNNER_SHA256)
from medrag.training.graph_input_data import ensure_contract


class PairMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.hidden = torch.randn(1, 10, 16)
        self.spans = [[1, 3], [5, 8]]
        self.memory = [{"receiver": 0, "sender": 1, "states": torch.randn(3, 8)},
                       {"receiver": 1, "sender": 0, "states": torch.randn(2, 8)}]
        self.bridge = PairMemoryBridge(16, 8, 8, 2)

    def test_zero_and_only_proposition_writes(self):
        messages = self.bridge.messages(self.hidden, self.spans, self.memory)
        self.assertTrue(torch.equal(apply_messages(self.hidden, self.spans, messages, torch.ones(2)), self.hidden))
        with torch.no_grad():
            self.bridge.o.weight.normal_(std=.03)
        messages = self.bridge.messages(self.hidden, self.spans, self.memory)
        self.assertTrue(torch.equal(apply_messages(self.hidden, self.spans, messages, torch.zeros(2)), self.hidden))
        result = apply_messages(self.hidden, self.spans, messages, torch.tensor([1., 0.]))
        # Early proposition can read later proposition through additional memory.
        self.assertFalse(torch.equal(result[:, 1:3], self.hidden[:, 1:3]))
        self.assertTrue(torch.equal(result[:, :1], self.hidden[:, :1]))
        self.assertTrue(torch.equal(result[:, 3:], self.hidden[:, 3:]))

    def test_joint_gate_gradients(self):
        with torch.no_grad():
            self.bridge.o.weight.normal_(std=.03)
        messages = [(r, i, x.detach()) for r, i, x in self.bridge.messages(self.hidden, self.spans, self.memory)]
        gate = torch.ones(2, requires_grad=True)
        output = apply_messages(self.hidden, self.spans, messages, gate)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertTrue((gate.grad.abs() > 0).all())

    def test_native_llama_and_backward(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                 num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                 max_position_embeddings=64, attention_dropout=0.)
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).eval()
        wrapper = FrozenLlamaMemory(model, cut=1)
        row = {"input_ids": list(range(10)), "choice_token_ids": [10, 11, 12, 13]}
        prefix = wrapper.prefix(row)
        with torch.no_grad():
            native = model(torch.tensor([row["input_ids"]]), use_cache=False).logits[0, -1, [10, 11, 12, 13]]
            plain = wrapper.logits(row, prefix)
        torch.testing.assert_close(native, plain)
        messages = self.bridge.messages(prefix, self.spans, self.memory)
        value = apply_messages(prefix, self.spans, messages, torch.ones(2))
        answer_loss(wrapper.logits(row, value), 0).backward()
        self.assertGreater(float(self.bridge.o.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_reporting_and_gold_selector(self):
        b = [{"ce": 2., "p_gold": .1, "correct": False}, {"ce": .1, "p_gold": .9, "correct": True}]
        x = [{"ce": .2, "p_gold": .8, "correct": True}, b[1]]
        summary = stats(x, b)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["delta_accuracy_pp"], 50.)
        self.assertEqual(summary["wrong_to_correct"], 1)
        self.assertTrue(better(x[0], b[0]))

    def test_cached_oracle_does_not_count_as_fresh_work(self):
        class Progress:
            def update(self):
                raise AssertionError("Cached record inflated active throughput")
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "unit-test")
            row = {"sample_id": "cached"}
            store.save(store.path("oracle_kg", row, "json"), {"best": {"correct": True}})
            result = oracle(None, store, None, None, row, None, "kg", Progress())
            self.assertTrue(result["best"]["correct"])

    def test_baseline_audit_records_near_tie_without_changing_current_baseline(self):
        original = {"sample_id": "x", "base": {"gold": 1, "prediction": 1,
                    "correct": True, "margin": .02, "logits": [0., .02, -.1, -.2]}}
        current = {"sample_id": "x", "dataset": "medqa", "base": {"gold": 1,
                   "prediction": 0, "correct": False, "margin": -.01,
                   "logits": [0., -.01, -.1, -.2]}}
        audit = baseline_audit([current], {"x": original})
        self.assertEqual(audit["changed_predictions"], 1)
        self.assertEqual((audit["old_correct"], audit["native_correct"]), (1, 0))
        self.assertEqual(audit["differences"][0]["sample_id"], "x")

    def test_only_original_runner_hash_migrates_cached_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            old = {"code": {RUNNER: INITIAL_RUNNER_SHA256, "other": "fixed"}, "dataset": "fixed"}
            h = ensure_contract(path, old)
            patched = copy.deepcopy(old)
            patched["code"][RUNNER] = "reviewed-patch"
            self.assertEqual(ensure_resume_contract(path, patched), h)
            self.assertTrue(path.with_name("BASELINE_AUDIT_PATCH.json").exists())
            self.assertEqual(ensure_resume_contract(path, patched), h)
            future = copy.deepcopy(patched)
            future["code"][RUNNER] = "unreviewed-future-patch"
            with self.assertRaises(RuntimeError):
                ensure_resume_contract(path, future)
            changed_data = copy.deepcopy(patched)
            changed_data["dataset"] = "different"
            with self.assertRaises(RuntimeError):
                ensure_resume_contract(path, changed_data)


if __name__ == "__main__":
    unittest.main()

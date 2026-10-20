import unittest

import torch

from medrag.generation.pair_memory_attention import (
    PairMemoryBridge, FrozenLlamaMemory, apply_messages, answer_loss)
from scripts.pilot_rag2_pair_memory_oracle import stats, better


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


if __name__ == "__main__":
    unittest.main()

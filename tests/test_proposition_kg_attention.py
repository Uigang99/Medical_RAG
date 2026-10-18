"""Bounded CPU regression tests; no local checkpoint or GPU required."""
import copy
import unittest

import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.proposition_kg_attention import (
    PropositionAttentionModel, PropositionPairController, _AttentionContext,
    block_bias, validate_layout,
)


def row_fixture(extra=0):
    # Evidence 1: tokens 1:4; evidence 2: 5:8; evidence 3: 9:12.
    # Question STEM 13:16; all later tokens represent options/answer prefix.
    return {"input_ids": list(range(1, 21 + extra)),
            "proposition_spans": [[1, 4], [5, 8], [9, 12]],
            "question_span": [13, 16],
            "pairs": [[1, 0], [2, 0], [2, 1]],
            "pair_roles": [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]}


def fixture(checkpointing=False):
    torch.manual_seed(123)
    cfg = LlamaConfig(vocab_size=67, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, pad_token_id=0, attention_dropout=0.)
    cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(cfg).eval()
    ctrl = PropositionPairController(32, hidden=12, rounds=2, layers=2)
    wrapper = PropositionAttentionModel(model, ctrl, cut_layer=2, attention_layers=(2, 3),
                                        gradient_checkpointing=checkpointing, query_chunk=7)
    return wrapper


def score(wrapper, rows):
    output = wrapper(rows, 0)
    logits = F.linear(output.last_hidden, wrapper.llm.lm_head.weight[[31, 32, 33, 34]])
    return F.cross_entropy(logits.float(), torch.zeros(len(rows), dtype=torch.long)) + .001 * output.bias_penalty.mean()


class PropositionAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_bias_raw_llama_parity_and_padding(self):
        wrapper = fixture()
        rows = [row_fixture(), row_fixture(3)]
        expected = []
        for row in rows:
            with torch.no_grad():
                expected.append(wrapper.llm.model(torch.tensor([row["input_ids"]]), use_cache=False).last_hidden_state[0, -1])
        output = wrapper(rows, 0)
        torch.testing.assert_close(output.last_hidden, torch.stack(expected), atol=2e-6, rtol=2e-5)
        self.assertEqual(output.bias_penalty.tolist(), [0., 0.])
        for item in output.biases:
            self.assertEqual(tuple(item.shape), (3, 2))
        wrapper.controller.enabled = False
        disabled = wrapper(rows, 0)
        torch.testing.assert_close(disabled.last_hidden, output.last_hidden, atol=2e-6, rtol=2e-5)

    def test_controller_gradient_and_frozen_base(self):
        wrapper = fixture()
        wrapper.train()
        self.assertFalse(wrapper.llm.training)
        value = score(wrapper, [row_fixture()])
        value.backward()
        self.assertGreater(wrapper.controller.output.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in wrapper.llm.parameters()))
        optimizer = torch.optim.AdamW(wrapper.controller.parameters(), lr=.01)
        optimizer.step(); optimizer.zero_grad()
        score(wrapper, [row_fixture()]).backward()
        self.assertGreater(wrapper.controller.proposition_projection[0].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(wrapper.controller.messages[0][0].weight.grad.abs().sum().item(), 0.)

    def test_checkpoint_gradient_and_immutable_context(self):
        plain, checked = fixture(False), fixture(True)
        with torch.no_grad():
            plain.controller.output.weight.normal_(0, .1)
        checked.controller.load_state_dict(plain.controller.state_dict())
        plain.train(); checked.train()
        rows = [row_fixture(), row_fixture(2)]
        first = score(plain, rows)
        # Extra forward before backwards must not replace checkpoint context.
        second = score(checked, rows)
        score(checked, [row_fixture(4)])
        first.backward(); second.backward()
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        for (name, p), (_, q) in zip(plain.controller.named_parameters(), checked.controller.named_parameters(), strict=True):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_choices_and_gold_never_change_controller_input(self):
        wrapper = fixture()
        with torch.no_grad():
            wrapper.controller.output.weight.normal_(0, .2)
        row = row_fixture()
        original = wrapper([row], 0).biases[0]
        changed = copy.deepcopy(row)
        changed["answer"] = "D"
        changed["target_ids"] = [66]
        changed["input_ids"][16:] = [65, 64, 63, 62]
        other = wrapper([changed], 0).biases[0]
        torch.testing.assert_close(original, other, atol=0, rtol=0)
        changed["input_ids"][13] = 61
        new_question = wrapper([changed], 0).biases[0]
        self.assertGreater((original - new_question).abs().max().item(), 1e-5)

    def test_importance_has_no_receiver_specific_readout(self):
        wrapper = fixture()
        wrapper.controller.variant = "importance"
        with torch.no_grad():
            wrapper.controller.output.weight.normal_(0, .2)
        output = wrapper([row_fixture()], 0)
        # [1<-0] and [2<-0] must have identical bias despite different receiver/roles.
        torch.testing.assert_close(output.biases[0][0], output.biases[0][1], atol=0, rtol=0)

    def test_bias_block_locations_and_direction(self):
        tokens = torch.tensor([3, 0, 0, 3, 1, 1, 3, 2, 2, 3, 3, 3])
        matrix = torch.zeros(4, 4)
        matrix[1, 0] = .75
        matrix[2, 1] = -.5
        context = _AttentionContext(torch.tensor([12]), (tokens,), (matrix,), 3)
        actual = block_bias(context, 0, 12, torch.float32)[0, 0]
        expected = torch.zeros(12, 12)
        expected[4:6, 1:3] = .75
        expected[7:9, 4:6] = -.5
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.triu(1).abs().sum().item(), 0.)

    def test_no_edges_and_no_propositions_are_valid(self):
        wrapper = fixture(True)
        row = row_fixture()
        row.update(proposition_spans=[], pairs=[], pair_roles=[])
        output = wrapper([row], 0)
        self.assertEqual(tuple(output.biases[0].shape), (0, 2))
        self.assertTrue(output.bias_penalty.requires_grad)
        score(wrapper, [row]).backward()
        self.assertEqual(wrapper.controller.output.weight.grad.abs().sum().item(), 0.)

    def test_layout_rejects_unsafe_spans_and_duplicate_or_reverse_pairs(self):
        for changes in [
            {"question_span": [13, 21]},
            {"proposition_spans": [[1, 4], [4, 14]]},
            {"pairs": [[0, 1]], "pair_roles": [[1, 0, 0, 0]]},
            {"pairs": [[1, 0], [1, 0]], "pair_roles": [[1, 0, 0, 0]] * 2},
        ]:
            row = {**row_fixture(), **changes}
            with self.assertRaises(ValueError):
                validate_layout(row)


if __name__ == "__main__":
    unittest.main()

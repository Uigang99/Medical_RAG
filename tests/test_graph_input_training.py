from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from medrag.training.graph_input_data import (duplicate_groups, split_questions, graph_from_edges,
                                             graph_stats, pack_pair, ensure_contract)
from medrag.training.graph_input_modules import GraphPrompt, TextPrompt, answer_losses


def edge(pid, s="A", o="B", relation="inhibits"):
    return {"edge_id": "edge:"+pid, "proposition_id": pid, "document_id": "doc:"+pid,
            "subject_entity_id": s, "object_entity_id": o,
            "subject_surface": s, "object_surface": o,
            "relation_surface": relation, "claim": f"{s} {relation} {o}."}


class Tokenizer:
    unk_token_id = None

    def apply_chat_template(self, messages, **kw):
        return [1] + [2] * len(" ".join(m["content"] for m in messages).split())

    def encode(self, text, **kw):
        return [3] * len(text.split())

    def convert_tokens_to_ids(self, token):
        return 4


class DataTests(unittest.TestCase):
    def test_groups_stay_together_and_no_outcome_selection(self):
        rows = [{"sample_id": str(i), "dataset": "medqa" if i % 3 else "medmcqa",
                 "question": f"Unique clinical vignette number {i}", "answer": "A"} for i in range(100)]
        rows[1]["question"] = rows[2]["question"]
        sizes = dict(train=40, validation=10, pilot_evaluation=10, reserved=40)
        one, report = split_questions(rows, sizes, 42)
        two, _ = split_questions([{**r, "answer": "D"} for r in rows], sizes, 42)
        self.assertEqual(one, two)
        by_id = {r["sample_id"]: r for r in one}
        self.assertEqual(by_id["1"]["split"], by_id["2"]["split"])
        self.assertEqual(sum(report["actual"].values()), 100)
        self.assertFalse(any("answer" in r or "question" in r for r in one))

    def test_near_duplicate(self):
        words = [f"word{i}" for i in range(100)]
        modified = words.copy()
        modified[-1] = "replacement"
        groups, _ = duplicate_groups([{"sample_id": "a", "question": " ".join(words)},
                                       {"sample_id": "b", "question": " ".join(modified)}])
        self.assertEqual(len(groups), 1)

    def test_direction_identity_and_disconnected_components(self):
        g = graph_from_edges([edge("1"), edge("2", "B", "C"), edge("3", "D", "E")])
        self.assertEqual(g["edge_index"], [[0, 1, 3], [1, 2, 4]])
        self.assertEqual(graph_stats(g)["components"], 2)
        self.assertTrue(graph_stats(g)["directed_two_hop"])
        self.assertEqual(graph_stats(g)["largest_component_edges"], 2)

    def test_joint_packing_and_no_gold_in_prompt(self):
        row = {"sample_id": "q", "question": "Question?", "options": {"A": "X", "B": "Y"}, "answer": "A"}
        edges = [edge(str(i)) for i in range(20)]
        a = pack_pair(Tokenizer(), row, edges, 70, 10)
        b = pack_pair(Tokenizer(), {**row, "answer": "B"}, edges, 70, 10)
        self.assertEqual(a["text_prompt_ids"], b["text_prompt_ids"])
        self.assertEqual(a["graph"], b["graph"])
        self.assertGreater(a["packing"]["removed_propositions"], 0)
        self.assertEqual({e["proposition_id"] for e in a["edges"]},
                         {p["proposition_id"] for p in a["propositions"]})

    def test_contract_refuses_silent_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            ensure_contract(path, {"a": 1})
            ensure_contract(path, {"a": 1})
            with self.assertRaises(RuntimeError):
                ensure_contract(path, {"a": 2})

    def test_chat_template_container_is_unwrapped(self):
        class NewTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kw):
                self.assert_return_dict = kw["return_dict"]
                return {"input_ids": super().apply_chat_template(messages, **kw)}
        tok = NewTokenizer()
        row = {"sample_id": "q", "question": "Question?", "options": {"A": "X"}, "answer": "A"}
        item = pack_pair(tok, row, [edge("1")], 8192, 10)
        self.assertIsInstance(item["text_prompt_ids"], list)
        self.assertFalse(tok.assert_return_dict)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_graph_permutation_invariance_and_direction_sensitivity(self):
        torch.manual_seed(1)
        m = GraphPrompt(8, 16, 12, 2, 3).eval()
        x, attrs = torch.randn(4, 8), torch.randn(3, 8)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
        expected = m(x, ei, attrs)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        actual = m(x[permutation], inverse[ei], attrs)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertFalse(torch.allclose(expected, m(x, ei.flip(0), attrs)))
        self.assertEqual(m(x[:0], ei[:, :0], attrs[:0]).shape, (1, 16))

    def test_frozen_llama_both_arms_forward_backward_and_exact_loss(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(5)
        llm = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.))
        llm.requires_grad_(False)
        llm.eval()
        llm.config.use_cache = False
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.model.train()
        emb = llm.get_input_embeddings()
        rows = [{"text_prompt_ids": [1, 2, 3], "graph_prompt_ids": [1, 4], "target_ids": [5, 6]},
                {"text_prompt_ids": [1, 7], "graph_prompt_ids": [1, 8, 9], "target_ids": [7, 6]}]
        for arm in ("text", "graph"):
            if arm == "text":
                adapter = TextPrompt(emb(torch.tensor([11, 12])))
                soft = [adapter(), adapter()]
            else:
                adapter = GraphPrompt(8, 32, 12, 2, 3)
                soft = [adapter(torch.randn(3, 8), torch.tensor([[0, 1], [1, 2]]), torch.randn(2, 8)) for _ in rows]
            losses = answer_losses(llm, emb, rows, soft, arm, 0)
            self.assertTrue(torch.isfinite(losses).all())
            # Compare target-only logits optimization to full causal LM loss.
            r, z = rows[0], soft[0]
            ids = r[arm+"_prompt_ids"] + r["target_ids"]
            e = emb(torch.tensor(ids))
            full = torch.cat((e[:1], z, e[1:]))[None]
            labels = torch.tensor([[-100]*(len(r[arm+"_prompt_ids"])+len(z)) + r["target_ids"]])
            reference = llm(inputs_embeds=full, labels=labels).loss
            torch.testing.assert_close(losses[0], reference, atol=1e-5, rtol=1e-5)
            losses.mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()))
            self.assertTrue(all(p.grad is None for p in llm.parameters()))
            before = [p.detach().clone() for p in adapter.parameters()]
            torch.optim.AdamW(adapter.parameters(), lr=.001).step()
            self.assertTrue(any(not torch.equal(a, p) for a, p in zip(before, adapter.parameters())))
            copy_adapter = copy.deepcopy(adapter)
            copy_adapter.load_state_dict(adapter.state_dict())

    def test_entrypoint_has_no_generation_or_evaluation_mode(self):
        source = (Path(__file__).parents[1] / "scripts/train_rag2_text_kg_input_pilot.py").read_text()
        self.assertNotIn(".generate(", source)
        self.assertNotIn('choices=("train", "test")', source)


if __name__ == "__main__":
    unittest.main()

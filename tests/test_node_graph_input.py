from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from medrag.training.graph_input_data import graph_from_edges
from medrag.training.graph_input_modules import answer_losses, training_attention_backend
from medrag.training.node_graph_input import NodeGraphPrompt, node_prompt_batch
from train_rag2_node_graph_input import arguments, validate_rows


def row(sid="q", split="train"):
    edges = [{"edge_id": "e1", "document_id": "d0", "proposition_id": "p1",
              "subject_entity_id": "A", "subject_surface": "A",
              "object_entity_id": "B", "object_surface": "B", "relation_surface": "inhibits"},
             {"edge_id": "e2", "document_id": "d1", "proposition_id": "p2",
              "subject_entity_id": "B", "subject_surface": "B",
              "object_entity_id": "C", "object_surface": "C", "relation_surface": "activates"}]
    return {"sample_id": sid, "split": split, "document_ids": [f"d{i}" for i in range(8)],
            "graph": graph_from_edges(edges), "edges": edges,
            "propositions": [{"proposition_id": "p1"}, {"proposition_id": "p2"}],
            "graph_prompt_ids": [1, 2, 3], "text_prompt_ids": [1, 3, 2, 3], "target_ids": [4, 5]}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(), row("v", "validation")]
        self.assignments = [{"sample_id": r["sample_id"], "split": r["split"],
                             "duplicate_group": r["sample_id"]} for r in self.rows]
        self.assignments.append({"sample_id": "test", "split": "pilot_evaluation", "duplicate_group": "test"})

    def test_frozen_scope_context_and_cross_document_identity(self):
        s = validate_rows(self.rows, self.assignments, 8)
        self.assertEqual(s["max_graph_context_with_target"], 8)
        self.assertEqual(s["counts"], {"train": 1, "validation": 1})
        self.assertEqual(self.rows[0]["graph"]["node_ids"], ["A", "B", "C"])
        self.assertEqual(self.rows[0]["graph"]["edge_index"], [[0, 1], [1, 2]])

    def test_overflow_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(RuntimeError, "context overflow"):
            validate_rows(self.rows, self.assignments, 7)

    def test_heldout_or_duplicate_rows_fail(self):
        with self.assertRaisesRegex(RuntimeError, "held-out"):
            validate_rows(self.rows + [row("test", "pilot_evaluation")], self.assignments, 100)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_rows(self.rows + [self.rows[0]], self.assignments, 100)

    def test_graph_mutation_or_extra_evidence_fails(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["graph"]["edge_index"][0][0] = 2
        with self.assertRaisesRegex(RuntimeError, "frozen edges"):
            validate_rows(changed, self.assignments, 100)
        changed = copy.deepcopy(self.rows)
        changed[0]["edges"][0]["document_id"] = "new_document"
        with self.assertRaisesRegex(RuntimeError, "outside Top-8"):
            validate_rows(changed, self.assignments, 100)

    def test_output_roots_separate_target_direction_and_smoke(self):
        roots = {arguments(flags).output_root for flags in
                 ([], ["--target-llm", "base"], ["--message-direction", "incoming"], ["--smoke-steps", "2"])}
        self.assertEqual(len(roots), 4)
        self.assertTrue(all("rag2_node_graph_input_v1" in str(p) for p in roots))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_node_alignment_permutation_equivariance_and_empty(self):
        torch.manual_seed(7)
        model = NodeGraphPrompt(8, 16, 12, 2, 3).eval()
        x, e = torch.randn(4, 8), torch.randn(3, 8)
        index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        z = model(x, index, e)
        self.assertEqual(z.shape, (4, 16))
        order = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(order)
        torch.testing.assert_close(model(x[order], inverse[index], e), z[order], atol=2e-6, rtol=2e-6)
        self.assertFalse(torch.allclose(z[0], z[1]))
        self.assertEqual(model(x[:0], index[:, :0], e[:0]).shape, (0, 16))
        self.assertEqual(model(x[:1], index[:, :0], e[:0]).shape, (1, 16))

    def test_subject_receives_outgoing_information_without_inverse_facts(self):
        torch.manual_seed(9)
        x, e = torch.randn(2, 8), torch.randn(1, 8)
        index = torch.tensor([[0], [1]])
        original_index = index.clone()
        changed = x.clone(); changed[1] += 2.
        incoming = NodeGraphPrompt(8, 16, 12, 1, 3, bidirectional=False).eval()
        both = NodeGraphPrompt(8, 16, 12, 1, 3).eval()
        torch.testing.assert_close(incoming(x, index, e)[0], incoming(changed, index, e)[0])
        self.assertFalse(torch.allclose(both(x, index, e)[0], both(changed, index, e)[0]))
        self.assertFalse(torch.allclose(both(x, index, e), both(x, index.flip(0), e)))
        self.assertFalse(torch.allclose(both(x, index, e), both(x, index, -e)))
        torch.testing.assert_close(index, original_index)

    def test_cross_document_two_hop_and_disconnected_scope(self):
        torch.manual_seed(12)
        g = row()["graph"]
        index = torch.tensor(g["edge_index"])
        x, e = torch.randn(4, 8), torch.randn(2, 8)
        one = NodeGraphPrompt(8, 16, 12, 1, 3).eval()
        two = NodeGraphPrompt(8, 16, 12, 2, 3).eval()
        changed = x.clone(); changed[2] += 3.
        torch.testing.assert_close(one(x, index, e)[0], one(changed, index, e)[0])
        self.assertFalse(torch.allclose(two(x, index, e)[0], two(changed, index, e)[0]))
        torch.testing.assert_close(two(x, index, e)[3], two(changed, index, e)[3])

    def test_feature_batch_preserves_node_order_and_does_not_mix_rows(self):
        torch.manual_seed(3)
        model = NodeGraphPrompt(8, 16, 12, 2, 3).eval()
        rows = [row(), row("other")]
        texts = ["A", "B", "C", "inhibits", "activates"]
        features = torch.randn(5, 8); lookup = {s: i for i, s in enumerate(texts)}
        z = node_prompt_batch(model, rows, features, lookup, "cpu")
        self.assertEqual([t.shape for t in z], [(3, 16), (3, 16)])
        torch.testing.assert_close(z[0], z[1])

    @training_attention_backend()
    def test_variable_node_tokens_frozen_lora_gradients_and_loss_alignment(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(14)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=32,
                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.))
        wrapper = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                                lora_dropout=0., task_type="CAUSAL_LM"))
        llm = wrapper.get_base_model()
        llm.requires_grad_(False); llm.eval(); llm.config.use_cache = False
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.model.train()
        adapter = NodeGraphPrompt(8, 24, 12, 2, 3)
        before_llm = {k: v.detach().clone() for k, v in llm.named_parameters()}
        rows = [{"graph_prompt_ids": [1, 2, 3], "target_ids": [4, 5]},
                {"graph_prompt_ids": [1, 3], "target_ids": [6, 5]},
                {"graph_prompt_ids": [1, 2], "target_ids": [7, 5]}]
        soft = [adapter(torch.randn(3, 8), torch.tensor([[0, 1], [1, 2]]), torch.randn(2, 8)),
                adapter(torch.randn(1, 8), torch.empty(2, 0, dtype=torch.long), torch.empty(0, 8)),
                adapter(torch.empty(0, 8), torch.empty(2, 0, dtype=torch.long), torch.empty(0, 8))]
        emb = llm.get_input_embeddings()
        losses = answer_losses(llm, emb, rows, soft, "graph", 0)
        self.assertTrue(torch.isfinite(losses).all())
        for r, z, value in zip(rows, soft, losses):
            e = emb(torch.tensor(r["graph_prompt_ids"] + r["target_ids"]))
            inputs = torch.cat((e[:1], z, e[1:]))[None]
            labels = torch.tensor([[-100]*(len(r["graph_prompt_ids"])+len(z)) + r["target_ids"]])
            torch.testing.assert_close(value, llm(inputs_embeds=inputs, labels=labels).loss, atol=1e-5, rtol=1e-5)
        losses.mean().backward()
        self.assertTrue(all(p.grad is None for p in llm.parameters()))
        self.assertGreater(adapter.layers[0].receiver_role.grad.abs().sum().item(), 0.)
        self.assertGreater(adapter.projector[0].weight.grad.abs().sum().item(), 0.)
        torch.optim.AdamW(adapter.parameters(), lr=.01).step()
        for k, p in llm.named_parameters(): torch.testing.assert_close(p, before_llm[k])

    @training_attention_backend()
    def test_tiny_engineering_overfit_not_medical_quality_claim(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(23)
        llm = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=24, intermediate_size=32,
                   num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.))
        llm.requires_grad_(False); llm.eval()
        adapter = NodeGraphPrompt(8, 24, 12, 1, 3)
        x, attrs, index = torch.randn(3, 8), torch.randn(2, 8), torch.tensor([[0, 1], [1, 2]])
        rows = [{"graph_prompt_ids": [1, 2], "target_ids": [3]}]
        opt = torch.optim.AdamW(adapter.parameters(), lr=.03)
        values = []
        for _ in range(45):
            opt.zero_grad(set_to_none=True)
            z = adapter(x, index, attrs)
            loss = answer_losses(llm, llm.get_input_embeddings(), rows, [z], "graph", 0).mean()
            values.append(loss.item()); loss.backward(); opt.step()
        self.assertLess(values[-1], values[0] - .05)
        self.assertTrue(all(p.grad is None for p in llm.parameters()))

    def test_no_generation_or_old_checkpoint_reuse(self):
        source = (ROOT / "scripts/train_rag2_node_graph_input.py").read_text()
        self.assertNotIn(".generate(", source)
        self.assertNotIn("evaluation_pairs.jsonl", source)
        self.assertNotIn("rag2_text_kg_input_training_v1_sdpa_fix/graph/best.pt", source)


if __name__ == "__main__":
    unittest.main()

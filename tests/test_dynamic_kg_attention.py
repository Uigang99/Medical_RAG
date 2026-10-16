import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_kg_structure_attention import row_fixture, tiny_model
from medrag.generation.dynamic_kg_attention import DynamicBias, source_layout
from medrag.generation.kg_structure_attention import answer_losses, token_layout
from medrag.training.dynamic_kg_features import FeatureBank, article_texts, feature_plan, token_windows
from medrag.text_graph_evaluation import generate_tokens


def fixture():
    torch.manual_seed(12)
    spec = {"sample_id": "x", "question_text": "query", "node_texts": ["A", "B", "C"],
            "proposition_texts": ["A inhibits B", "B activates C"],
            "edges": [[0, 1, 0, "inhibits"], [1, 2, 1, "activates"]],
            "links": [[1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, -1]],
            "pairs": [[0, 4, 2], [1, 3, 1]], "pair_links": [[0, 1], [1, 0]]}
    texts = list(dict.fromkeys(article_texts(spec)))
    vectors = {"query": torch.randn(3, 8), "article": torch.randn(len(texts), 8)}
    index = {"query": {"query": (0, 2), "other": (2, 3)},
             "article": {t: (i, i+1) for i, t in enumerate(texts)}}
    return spec, FeatureBank(vectors, index)


def controller_fixture():
    spec, bank = fixture()
    ctrl = DynamicBias(2, 4, hidden=12, rounds=2, input_dim=8, query_chunk=2)
    ctrl.bind_bank(bank)
    return ctrl, spec


class DynamicTests(unittest.TestCase):
    def test_exact_mapping_dedup_and_gold_independence(self):
        tok, row, _ = row_fixture()
        for i, edge in enumerate(row["edges"]): edge["relation_surface"] = ["inhibits", "activates"][i]
        output = source_layout(tok, row)
        self.assertEqual(output["pairs"], token_layout(tok, row)["pairs"])
        other = copy.deepcopy(row); other["answer"] = "Z"; other["target_ids"] = [999]
        self.assertEqual(output, source_layout(tok, other))
        row["edges"].append(copy.deepcopy(row["edges"][0]))
        self.assertEqual(output["links"], source_layout(tok, row)["links"])
        self.assertEqual(output["pair_links"], source_layout(tok, row)["pair_links"])
        plan = feature_plan([output])
        self.assertEqual(plan["query"], [row["question"]])
        self.assertNotIn("options", output)

    def test_ambiguous_and_missing_mapping(self):
        tok, row, _ = row_fixture()
        for e in row["edges"]: e["relation_surface"] = "relates"
        row["edges"][0]["subject_surface"] = "ABSENT"
        d = copy.deepcopy(row["edges"][1]); d["subject_entity_id"] = "NOT_B"
        row["edges"].append(d)
        output = source_layout(tok, row)
        self.assertEqual(output["pairs"], token_layout(tok, row)["pairs"])
        self.assertEqual(output["statistics"]["identity_ambiguous_mentions"], 2)

    def test_windows_retain_all_tokens(self):
        class Tokenizer:
            cls_token_id, sep_token_id = -1, -2
            def __call__(self, text, **kw): return {"input_ids": list(range(len(text)))}
            def num_special_tokens_to_add(self, pair=False): return 2
        windows = token_windows(Tokenizer(), "x"*1060)
        self.assertEqual([len(w) for w in windows], [512, 512, 42])
        self.assertEqual([i for w in windows for i in w[1:-1]], list(range(1060)))
        self.assertEqual(token_windows(Tokenizer(), ""), [[-1, -2]])

    def test_installed_medcpt_tokenizer_window_and_padding_api(self):
        from transformers import AutoTokenizer
        path = Path("/home/user/Uiheon/models/MedCPT-Query-Encoder")
        if not path.exists(): self.skipTest("Local MedCPT tokenizer unavailable")
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        text = "A long medical question with known clinical findings."
        self.assertEqual(token_windows(tok, text)[0], tok(text)["input_ids"])
        long_text = text*140
        windows = token_windows(tok, long_text)
        self.assertEqual([t for w in windows for t in w[1:-1]], tok(long_text, add_special_tokens=False)["input_ids"])
        padded = tok.pad({"input_ids": windows[:2]}, padding=True, return_tensors="pt")
        self.assertIn("attention_mask", padded)
        self.assertLessEqual(padded["input_ids"].shape[1], 512)

    def test_zero_parity_gradient_checkpoint_and_live_input_dependence(self):
        model = tiny_model(); ctrl, spec = controller_fixture()
        rows = [{"text_prompt_ids": [1, 3, 4, 5, 6], "target_ids": [9, 10]}]
        expected = answer_losses(model, ctrl, rows, [spec], 0)
        ctrl.attach(model)
        loss = answer_losses(model, ctrl, rows, [spec], 0)
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        grad = ctrl.output.weight.grad.clone()
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        ctrl.zero_grad()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
        answer_losses(model, ctrl, rows, [spec], 0).mean().backward()
        torch.testing.assert_close(ctrl.output.weight.grad, grad)
        optimizer = torch.optim.AdamW(ctrl.parameters(), lr=.02, weight_decay=0)
        initial = loss.item()
        for _ in range(16):
            optimizer.zero_grad()
            value = answer_losses(model, ctrl, rows, [spec], 0).mean()
            value.backward(); optimizer.step()
        self.assertLess(value.item(), initial)
        self.assertGreater(ctrl.query_projection[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(ctrl.messages[0][0].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            v = ctrl.link_biases(spec)
            changed = {**spec, "question_text": "other"}
            self.assertGreater((v-ctrl.link_biases(changed)).abs().max().item(), 1e-5)
            ctrl.variant = "local"
            self.assertGreater((v-ctrl.link_biases(spec)).abs().max().item(), 1e-5)
            self.assertLessEqual(v.abs().max().item(), 2.)

    def test_cached_decode_padding_and_no_links(self):
        model = tiny_model(); ctrl, spec = controller_fixture()
        prompts = [[1, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]]
        expected = generate_tokens(model, prompts, pad_id=0, eos_ids=[2], max_new_tokens=3)
        ctrl.attach(model)
        ctrl.set_layouts([spec, spec], [1, 0])
        self.assertEqual(generate_tokens(model, prompts, pad_id=0, eos_ids=[2], max_new_tokens=3), expected)
        with torch.no_grad(): ctrl.output.weight.normal_(0, .1)
        ctrl.set_layouts([spec])
        ids = torch.tensor([prompts[0]])
        with torch.no_grad():
            pref = model(ids, use_cache=True)
            full = model(torch.cat((ids, torch.tensor([[12]])), 1)).logits[:, -1]
            cached = model(torch.tensor([[12]]), past_key_values=pref.past_key_values,
                           attention_mask=torch.ones(1, 6), use_cache=True).logits[:, -1]
        torch.testing.assert_close(cached, full, atol=1e-6, rtol=1e-5)
        empty = {**spec, "pairs": [], "pair_links": []}
        ctrl.set_layouts([empty])
        with torch.no_grad(): result = model(ids).logits
        ctrl.enabled = False
        with torch.no_grad(): expected = model(ids).logits
        torch.testing.assert_close(result, expected)

    def test_duplicate_link_memberships_average_not_sum(self):
        ctrl, spec = controller_fixture()
        with torch.no_grad(): ctrl.output.weight.normal_(0, .1)
        ctrl.set_layouts([spec]); expected = ctrl.biases[0].clone()
        duplicate = copy.deepcopy(spec)
        duplicate["links"].append(duplicate["links"][0])
        duplicate["pair_links"].append([1, 2])
        ctrl.set_layouts([duplicate])
        torch.testing.assert_close(ctrl.biases[0], expected)

    def test_training_entry_resume_after_last_batch_before_validation(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_rag2_dynamic_kg_attention.py"
        modspec = importlib.util.spec_from_file_location("dynamic_training_test", path)
        runner = importlib.util.module_from_spec(modspec); modspec.loader.exec_module(runner)
        class Progress:
            def stage(self, *args, **kwargs): pass
            def complete(self, *args): pass
            def update(self, *args): pass
            def log(self, *args): pass
        ctrl, spec = controller_fixture(); model = tiny_model()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        rows = [{"sample_id": str(i), "split": "train" if i < 4 else "validation",
                 "text_prompt_ids": [1, 3, 4, 5, 6], "target_ids": [9, 10]} for i in range(6)]
        c = {"run_hash": "test", "rows": rows, "layouts": {r["sample_id"]: spec for r in rows},
             "tokenizer": SimpleNamespace(eos_token_id=2)}
        with tempfile.TemporaryDirectory() as directory:
            a = SimpleNamespace(output_root=Path(directory), learning_rate=.01, epochs=1, seed=42,
                                micro_batch_size=1, effective_batch_size=2, checkpoint_steps=1)
            def factory(*args):
                fresh, _ = controller_fixture()
                return fresh
            original = runner.validation
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2: raise RuntimeError("simulated validation interruption")
                return original(*args, **kwargs)
            with patch.object(runner, "new_controller", side_effect=factory), \
                 patch.object(runner, "rng_state", return_value={}), \
                 patch.object(runner, "restore_rng"), \
                 patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                 patch.object(torch.cuda, "empty_cache"):
                with patch.object(runner, "validation", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        runner.full_train(a, c, model, "graph", Progress(), 0)
                runner.full_train(a, c, model, "graph", Progress(), 0)
                done = runner.json.loads((Path(directory) / "graph/full/COMPLETE.json").read_text())
                self.assertEqual(done["train_questions"], 4)
                self.assertEqual(done["evaluation_questions_used"], 0)
                self.assertFalse(done["tiny_weights_reused"])


if __name__ == "__main__": unittest.main()

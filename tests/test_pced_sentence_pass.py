from types import SimpleNamespace
import unittest

import torch

from medrag.generation import pced_sentence_pass as m
from medrag.generation import pced_pass_bounded as old


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[5])
    def __init__(self, outputs):
        self.outputs, self.inputs = outputs, []
    def __call__(self, input_ids, **kwargs):
        output = self.outputs[len(self.inputs)]
        self.inputs.append(input_ids.clone())
        return SimpleNamespace(logits=output[:, None, :])


class SentencePassTests(unittest.TestCase):
    def test_exclude_and_fixed_log_weight(self):
        raw = torch.tensor([[9., 1., 0.], [8., 0., 1.], [20., 0., 0.]])
        a = torch.tensor([0., 3., 0.])
        c = [.51, .95, .1]
        _, x = m.select(raw, a, raw, c, "sentence_exclude")
        _, y = m.select(raw, a, raw, c, "sentence_weighted")
        self.assertEqual(x["winner"], 0)
        self.assertEqual(y["winner"], 1)
        self.assertEqual(x["excluded_experts"], [2])
        self.assertAlmostEqual(y["weighted_best_scores"][0], 9+4*float(torch.tensor(.51).log()), places=5)
        self.assertEqual(y["counterfactual_same_prefix"]["pced"]["winner"], 2)

    def test_all_pass_keeps_experts_excluded_and_masks_fallback_eos(self):
        raw = torch.tensor([[1., 2., 9.], [2., 1., 9.]])
        tok, trace = m.select(raw, torch.tensor([1., 3., 20.]), raw, [.5, .01],
                              "sentence_weighted", forbidden=(2,))
        self.assertEqual(tok, 1)
        self.assertEqual(trace["winner"], -1)
        self.assertEqual(trace["excluded_experts"], [0, 1])
        self.assertTrue(trace["amateur_fallback"])

    def test_zero_weight_is_exact_exclude_and_no_gate_matches_old_pced(self):
        torch.manual_seed(42)
        raw, a, scores = torch.randn(3, 8), torch.randn(8), torch.randn(3, 8)
        self.assertEqual(m.select(raw, a, scores, [.8, .2, .7], "sentence_exclude")[0],
                         m.select(raw, a, scores, [.8, .2, .7], "sentence_weighted", weight=0)[0])
        new = m.select(raw, a, scores, [1.]*3, "pced", forbidden=(6, 7))
        prior = old.select(raw, a, scores, (6,), "pced", forbidden=(7,))
        self.assertEqual((new[0], new[1]["winner"]), (prior[0], prior[1]["winner"]))
        self.assertEqual(new[1]["score"], prior[1]["score"])

    def test_sentence_boundaries(self):
        for text in ("A complete fact.", "A fact!\n", 'He said "Done."', "Why?"):
            self.assertTrue(m.sentence_end(text), text)
        for text in ("\n", "Dr.", "It is 3.", "U.S.", "e.g.", "An unfinished line\n", "3.14"):
            self.assertFalse(m.sentence_end(text), text)

    def test_gate_frozen_reentry_and_excluded_streams_consume_shared_tokens(self):
        outputs = [torch.tensor([[0.]*6, [0., 0., 9., 0., 0., 0.], [0., 8., 0., 0., 0., 0.]])]*3
        model = FakeModel(outputs)
        context = {"attention_mask": torch.ones(3, 2, dtype=torch.long)}
        suffix = {"input_ids": torch.ones(3, 1, dtype=torch.long),
                  "attention_mask": torch.ones(3, 1, dtype=torch.long)}
        calls = []
        def judge(text, phase):
            calls.append((text, phase))
            c = [.1, .9] if len(calls) == 1 else [.9, .1]
            return {"contribute_probability": c, "seconds": 0.}
        s = m.Stream(model, None, context, suffix, torch.ones(2), (4,),
                     lambda a, b: torch.zeros_like(b), "sentence_weighted", judge)
        s.begin_segment("", "reasoning", "start")
        self.assertEqual(s.step("reasoning", 0)[0], 1)
        self.assertEqual(s.step("reasoning", 1)[0], 1)
        self.assertEqual(len(calls), 1)
        s.begin_segment("A fact.", "reasoning", "sentence_end")
        self.assertEqual(s.step("reasoning", 2)[0], 2)
        self.assertEqual([x["winner"] for x in s.traces], [1, 1, 0])
        self.assertEqual(model.inputs[1].tolist(), [[1]]*3)
        self.assertEqual(model.inputs[2].tolist(), [[1]]*3)

    def test_sentence_and_answer_decisions_no_text_pollution(self):
        class Tokens:
            def decode(self, ids, **kwargs):
                return "".join({0: "A fact.", 1: "\n", 2: "Another fact.", 3: "\nAns", 4: "wer:", 5: " Paris", 6: ""}[i] for i in ids)
            def encode(self, text, **kwargs): return [9]
        class Scripted:
            def __init__(self):
                self.values = iter([(i, i == 6) for i in range(7)])
                self.traces, self.events, self.controls, self.permissions = [], [], [], []
            def begin_segment(self, text, phase, trigger): self.events.append((text, phase, trigger))
            def step(self, phase, local, *, allow_eos):
                self.permissions.append(allow_eos)
                return next(self.values)
            def force(self, ids): self.controls.append(ids)
        s = Scripted()
        result = m.generate(s, Tokens())
        self.assertEqual(result["prediction"], "Paris")
        self.assertEqual(result["reasoning"], "A fact.\nAnother fact.")
        self.assertEqual(s.events, [("", "reasoning", "start"), ("A fact.", "reasoning", "sentence_end"),
                                    ("A fact.\nAnother fact.", "reasoning", "sentence_end"),
                                    ("A fact.\nAnother fact.", "answer", "answer_start")])
        self.assertEqual(s.permissions, [False]*6+[True])
        self.assertEqual(s.controls, [])
        # Without a gate the new parser produces exactly the old natural CoT output.
        prior = old.generate(Scripted(), Tokens())
        for key in ("reasoning", "prediction", "reasoning_token_ids", "answer_token_ids", "reasoning_stop", "answer_status"):
            self.assertEqual(result[key], prior[key])

    def test_reasoning_cap_has_fresh_answer_decision_and_no_empty_eos(self):
        class Tokens:
            def decode(self, ids, **kwargs): return "".join("unfinished" if i == 0 else " " for i in ids)
            def encode(self, text, **kwargs): return [8, 9]
        class Scripted:
            def __init__(self): self.traces, self.events, self.controls = [], [], []
            def begin_segment(self, text, phase, trigger): self.events.append((text, phase))
            def step(self, phase, local, *, allow_eos):
                assert not allow_eos
                return (0 if phase == "reasoning" else 1), False
            def force(self, ids): self.controls.append(ids)
        s = Scripted()
        result = m.generate(s, Tokens(), reasoning_tokens=2, answer_tokens=2)
        self.assertEqual(s.controls, [[8, 9]])
        self.assertEqual([e[1] for e in s.events], ["reasoning", "answer"])
        self.assertTrue(result["empty_answer"])

    def test_v2_cap_preserves_unfinished_text_without_answer_or_repair(self):
        class Tokens:
            def decode(self, ids, **kwargs): return "A fact.\nAn unfinished connection"
            def encode(self, text, **kwargs): raise AssertionError("No synthetic header allowed")
        class Scripted:
            def __init__(self): self.traces, self.events = [], []
            def begin_segment(self, text, phase, trigger): self.events.append((text, phase))
            def step(self, phase, local, *, allow_eos):
                assert phase == "reasoning" and not allow_eos
                return 1, False
            def force(self, ids): raise AssertionError("Must not force Answer at budget")
        s = Scripted()
        result = m.generate(s, Tokens(), reasoning_tokens=2, answer_tokens=8, force_answer_at_budget=False)
        self.assertEqual(result["reasoning"], "A fact.\nAn unfinished connection")
        self.assertEqual(result["reasoning_token_ids"], [1, 1])
        self.assertEqual(result["answer_status"], "skipped_incomplete_reasoning")
        self.assertFalse(result["natural_answer_transition"])
        self.assertEqual(result["prediction"], "")
        self.assertEqual(result["forced_controls"], [])
        self.assertEqual(result["answer_token_ids"], [])
        self.assertTrue(all(e[1] == "reasoning" for e in s.events))
        from scripts.evaluate_loft_pced_sentence_pass import score_prediction
        self.assertEqual(score_prediction(result, ["A"]), 0)
        self.assertEqual(score_prediction(result, ["A fact"]), 0)

    def test_v2_natural_marker_at_last_budget_token_keeps_same_answer_stream(self):
        class Tokens:
            def decode(self, ids, **kwargs):
                return "".join({0: "A concrete fact.", 1: "\nAnswer:", 2: " Paris", 3: ""}[i] for i in ids)
        class Scripted:
            def __init__(self):
                self.traces, self.events, self.phases = [], [], []
                self.tokens = iter([(0, False), (1, False), (2, False), (3, True)])
            def begin_segment(self, text, phase, trigger): self.events.append((text, phase))
            def step(self, phase, local, *, allow_eos):
                self.phases.append((phase, allow_eos))
                return next(self.tokens)
            def force(self, ids): raise AssertionError("No forced text")
        s = Scripted()
        result = m.generate(s, Tokens(), reasoning_tokens=2, force_answer_at_budget=False)
        self.assertEqual(result["prediction"], "Paris")
        self.assertEqual(result["reasoning_stop"], "answer_marker")
        self.assertTrue(result["natural_answer_transition"])
        self.assertEqual(result["answer_status"], "eos")
        self.assertEqual(s.phases, [("reasoning", False), ("reasoning", False), ("answer", False), ("answer", True)])
        self.assertEqual(s.events[-1], ("A concrete fact.", "answer"))

    def test_v2_does_not_accept_inline_or_unfinished_answer_header(self):
        class Tokens:
            def __init__(self, text): self.text = text
            def decode(self, ids, **kwargs): return self.text
        class Scripted:
            traces, events = [], []
            def begin_segment(self, *args): pass
            def step(self, phase, local, *, allow_eos): return 1, False
            def force(self, ids): raise AssertionError("No forced text")
        for text in ("A comment about Answer: not a header.", "A fact.\nAns", "A fact.\nAnswer"):
            result = m.generate(Scripted(), Tokens(text), reasoning_tokens=1, force_answer_at_budget=False)
            self.assertEqual(result["answer_status"], "skipped_incomplete_reasoning")

    def test_v2_prompts_preserve_private_document_layout_and_shared_question(self):
        class Tokens:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]+"\nUSER:\n"+messages[1]["content"]+"\nASSISTANT:\n"
        prefixes, suffix = m.generation_prompts(Tokens(), "QUERY_SECRET", ["DOC_ALPHA", "DOC_BETA"])
        self.assertEqual(len(prefixes), 3)
        self.assertTrue(all(m.EVIDENCE_SYSTEM in p for p in prefixes))
        self.assertTrue(all("QUERY_SECRET" not in p for p in prefixes))
        self.assertIn("QUERY_SECRET", suffix)
        self.assertTrue(suffix.endswith(old.PREFIX))
        self.assertNotIn("DOC_ALPHA", prefixes[0]+prefixes[2]+suffix)
        self.assertNotIn("DOC_BETA", prefixes[0]+prefixes[1]+suffix)
        self.assertIn(old.AMATEUR_RULES, prefixes[0])
        self.assertNotIn(old.AMATEUR_RULES, prefixes[1])
        self.assertTrue(all(m.JUDGE_SYSTEM not in p for p in prefixes))
        with self.assertRaises(ValueError):
            m.generation_prompts(Tokens(), "<PCED_BOUNDARY_42>", ["text"])

    def test_v2_cli_has_separate_output_and_explicit_budget_override(self):
        from scripts.evaluate_loft_pced_sentence_pass import arguments
        legacy = arguments([])
        current = arguments(["--reasoning-style", "evidence_v2"])
        controlled = arguments(["--reasoning-style", "evidence_v2", "--reasoning-tokens", "128"])
        self.assertEqual(legacy.reasoning_tokens, 128)
        self.assertEqual(current.reasoning_tokens, 192)
        self.assertEqual(controlled.reasoning_tokens, 128)
        self.assertNotEqual(current.output_root, legacy.output_root)
        self.assertEqual(current.arms, list(m.ARMS))

    def test_real_cpu_llama_judge_cache_crop_matches_full_recompute(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(42)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=512, eos_token_id=31)).eval()
        class Tokens:
            pad_token_id = 0
            def encode(self, text, **kwargs): return [1+ord(c)%28 for c in text]
        tok = Tokens()
        prefixes = [[1, 2], [3, 4, 5]]
        judge = m.CachedJudge(model, tok, prefixes, "Decision:", (4, 5), 2)
        for text, phase in [("A fact.", "reasoning"), ("A fact. Another fact.", "answer"), ("", "reasoning")]:
            result = judge(text, phase)
            suffix = m.decision_suffix(tok, text, phase, "Decision:")
            ids = torch.tensor([[0, 1, 2]+suffix, [3, 4, 5]+suffix])
            mask = torch.tensor([[0, 1, 1]+[1]*len(suffix), [1]*(3+len(suffix))])
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1]
            expected = logits[:, [4, 5]].softmax(-1)[:, 0]
            torch.testing.assert_close(torch.tensor(result["contribute_probability"]), expected, atol=2e-6, rtol=2e-6)
            self.assertEqual(judge.groups[0][2].get_seq_length(), 3)


if __name__ == "__main__":
    unittest.main()

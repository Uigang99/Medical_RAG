from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from medrag.node_graph_evaluation import BASELINES, NODE_CONDITION, generation_inputs, generate_tokens, summarize
from medrag.text_graph_evaluation import CONDITIONS, parse_answer, summarize as old_summary
from medrag.training.graph_input_data import digest, graph_from_edges
from evaluate_rag2_node_graph_input import (arguments, preflight, read_prediction_shard, report,
                                          validate_node_cache, verify_rows)


def question(i):
    edges = [{"edge_id": "e", "proposition_id": "p", "document_id": "d0",
              "subject_entity_id": "a", "subject_surface": "A", "object_entity_id": "b",
              "object_surface": "B", "relation_surface": "inhibits"}]
    return {"sample_id": str(i), "dataset": "medmcqa", "answer": "A", "options": {"A": "x", "B": "y"},
            "document_ids": [f"d{j}" for j in range(8)], "edges": edges,
            "graph": graph_from_edges(edges), "graph_prompt_ids": [1, 2, 3]}


class NodeGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(12)
        return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=31, pad_token_id=0, attention_dropout=0.)).eval()

    def test_all_node_tokens_inserted_after_bos_and_empty_graph_supported(self):
        model = self.model()
        prompts = [[1, 2, 3], [1, 4], [1, 5, 6, 7]]
        soft = [torch.randn(3, 32), torch.randn(1, 32), torch.empty(0, 32)]
        with torch.inference_mode():
            packed = generation_inputs(model, prompts, soft, 0)
            self.assertEqual(packed["attention_mask"].tolist(), [[1]*6, [0]*3+[1]*3, [0]*2+[1]*4])
            torch.testing.assert_close(packed["inputs_embeds"][0, 1:4], soft[0])
            torch.testing.assert_close(packed["inputs_embeds"][1, 4:5], soft[1])
            torch.testing.assert_close(packed["inputs_embeds"][2, 2:], model.get_input_embeddings()(torch.tensor(prompts[2])))

    def test_generation_uses_node_embeddings_and_correctly_slices_variable_prefixes(self):
        model = self.model()
        prompts = [[1, 2, 3], [1, 4], [1, 5, 6, 7]]
        soft = [torch.randn(3, 32), torch.randn(1, 32), torch.empty(0, 32)]
        with torch.inference_mode():
            packed = generation_inputs(model, prompts, soft, 0)
            mask = packed["attention_mask"]
            first = model(inputs_embeds=packed["inputs_embeds"], attention_mask=mask,
                          position_ids=(mask.cumsum(-1)-1).clamp_min(0)).logits[:, -1].argmax(-1).tolist()
        actual = generate_tokens(model, prompts, soft, pad_id=0, eos_ids=[31], max_new_tokens=4)
        self.assertEqual([r["generated_ids"][0] for r in actual], first)
        for p, z, result in zip(prompts, soft, actual):
            single = generate_tokens(model, [p], [z], pad_id=0, eos_ids=[31], max_new_tokens=4)[0]
            self.assertEqual(result, single)
            self.assertLessEqual(len(result["generated_ids"]), 4)

    def test_bad_embedding_shape_or_values_fail(self):
        model = self.model()
        for z in (torch.randn(3, 31), torch.randn(32), torch.full((2, 32), float("nan"))):
            with self.assertRaises(ValueError): generation_inputs(model, [[1, 2]], [z], 0)

    def test_strict_parser_matches_existing_comparison_contract(self):
        self.assertEqual(parse_answer("Answer: B", {"A": "x", "B": "y"}), "B")
        self.assertIsNone(parse_answer("Answer: B or A", {"A": "x", "B": "y"}))
        self.assertIsNone(parse_answer("Answer: B. explanation", {"A": "x", "B": "y"}))


class ContractTests(unittest.TestCase):
    def test_same_512_scope_and_node_context_budget(self):
        rows = [question(i) for i in range(512)]
        assignments = [{"sample_id": str(i), "split": "pilot_evaluation"} for i in range(512)]
        stats = verify_rows(rows, assignments, 21, 16)
        self.assertEqual(stats["max_generation_context"], 21)
        with self.assertRaisesRegex(RuntimeError, "overflow"): verify_rows(rows, assignments, 20, 16)
        assignments[0]["split"] = "train"
        with self.assertRaisesRegex(RuntimeError, "SAME 512"): verify_rows(rows, assignments, 21, 16)

    def test_node_cache_requires_ids_order_scope_and_variable_length(self):
        rows = [question(0)]
        item = {"sample_id": "0", "node_ids": ["a", "b"], "graph_sha256": digest(rows[0]["graph"]),
                "values": torch.randn(2, 16)}
        validate_node_cache([item], rows, 16)
        for key, value in (("node_ids", ["b", "a"]), ("graph_sha256", "different"), ("values", torch.randn(1, 16))):
            with self.assertRaisesRegex(RuntimeError, "alignment/scope"):
                validate_node_cache([{**item, key: value}], rows, 16)

    def test_missing_completion_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(["--training-root", str(Path(tmp)/"incomplete"), "--output-root", str(Path(tmp)/"output")])
            with self.assertRaisesRegex(RuntimeError, "not complete"): preflight(args)

    def test_prediction_resume_is_contract_and_id_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"chunk.json"
            p.write_text(json.dumps({"contract_sha256": "h", "rows": [{"sample_id": "q"}]}))
            self.assertEqual(len(read_prediction_shard(p, ["q"], "h")), 1)
            with self.assertRaises(RuntimeError): read_prediction_shard(p, ["other"], "h")
            with self.assertRaises(RuntimeError): read_prediction_shard(p, ["q"], "other")


class ReportTests(unittest.TestCase):
    def fixtures(self):
        rows = [{"sample_id": str(i), "dataset": d, "answer": "A"} for i, d in
                enumerate(["medmcqa"]*3 + ["medqa"] + ["mmlu_anatomy"]*2)]
        pred = {}
        for name in BASELINES + (NODE_CONDITION,):
            values = ["A", "B", "B", "A", "A", "B"]
            if name == NODE_CONDITION: values = ["A", "A", "B", "A", "A", None]
            pred[name] = [{"sample_id": str(i), "prediction": a, "finish_reason": "eos"} for i, a in enumerate(values)]
        return rows, pred

    def test_metrics_and_paired_transitions_equal_previous_definitions(self):
        rows, pred = self.fixtures()
        result = summarize(rows, pred, 50)
        old = {name: pred[NODE_CONDITION] if name == "5_trained_graph" else pred["text"] for name, _, _ in CONDITIONS}
        previous = old_summary(rows, old, 50)
        self.assertEqual(result["conditions"][NODE_CONDITION], previous["conditions"]["5_trained_graph"])
        new_pair, old_pair = result["primary_comparison"], previous["primary_comparison"]
        for key in ("delta_micro_pp", "delta_macro3_pp", "transitions", "macro3_ci95_pp", "micro_ci95_pp"):
            self.assertEqual(new_pair[key], old_pair[key])
        self.assertEqual(new_pair["transitions"], {"C_C": 3, "C_W": 0, "W_C": 1, "W_W": 2})
        broken = copy.deepcopy(pred); broken[NODE_CONDITION].pop()
        with self.assertRaises(ValueError): summarize(rows, broken, 50)

    def test_summary_outputs_one_new_condition_and_three_reused_baselines(self):
        rows, pred = self.fixtures()
        class Progress:
            def stage(self, *a, **k): pass
            def update(self, *a, **k): pass
            def complete(self, *a, **k): pass
            def log(self, *a, **k): pass
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(["--output-root", tmp])
            context = {"rows": rows, "baselines": {n: pred[n] for n in BASELINES},
                       "contract": {"target_llm": "shared", "baselines": {}, "statistics": {}}}
            report(args, context, pred[NODE_CONDITION], "h", Progress())
            done = json.loads((Path(tmp)/"COMPLETE.json").read_text())
            self.assertEqual(done["new_conditions"], 1)
            self.assertEqual(done["cached_baseline_conditions"], 3)
            self.assertEqual(len((Path(tmp)/"paired_predictions.jsonl").read_text().splitlines()), 6)


if __name__ == "__main__": unittest.main()

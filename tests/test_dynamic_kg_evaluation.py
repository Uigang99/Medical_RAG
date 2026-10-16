import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from medrag.dynamic_kg_evaluation import paired_summary, read_shard, selected_controllers
from medrag.training.graph_input_data import atomic_json, digest, sha_file


def entry():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag2_dynamic_kg_attention.py"
    spec = importlib.util.spec_from_file_location("dynamic_eval_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class EvaluationTests(unittest.TestCase):
    def test_five_conditions_transitions_and_json(self):
        module = entry()
        rows = [{"sample_id": str(i), "dataset": d, "answer": "A"}
                for i, d in enumerate(("medmcqa", "medqa", "mmlu_anatomy"))]
        good = [{"sample_id": str(i), "prediction": "A", "finish_reason": "eos"} for i in range(3)]
        bad = [{**p, "prediction": "B"} for p in good]
        result = paired_summary(rows, {"no_rag": bad, "text": bad, "static_attention": bad,
                                       "local": bad, "graph": good}, module.base.summarize)
        self.assertEqual(len(result["conditions"]), 5)
        self.assertTrue(result["exploratory_gate"]["passed"])
        self.assertEqual(result["paired_comparisons"]["graph_vs_local"]["transitions"]["W_C"], 3)
        self.assertEqual(result["primary"]["condition"], "graph")
        json.dumps(result, allow_nan=False)

    def test_completed_checkpoint_required_and_selection_verified(self):
        manifest = {"contract_sha256": "hash", "contract": {"variants": ["graph"], "training": {"epochs": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                selected_controllers(root, manifest, ["graph"], lambda p: {})
            self.assertEqual(selected_controllers(root, manifest, ["graph"], lambda p: {}, allow_pending=True)[2], ["graph"])
            full = root / "graph/full"
            checkpoint = {"contract_sha256": "hash", "variant": "graph", "epoch": 1,
                          "validation_ce": .2, "controller": {"fake": 1}}
            atomic_json(full / "best.pt", checkpoint)
            atomic_json(full / "history.json", [{"epoch": 0, "validation_ce": .3, "validation_n": 256},
                {"epoch": 1, "validation_ce": .2, "validation_n": 256, "train_n": 2048}])
            atomic_json(root / "graph/tiny32/GATE.json", {"contract_sha256": "hash", "passed": True})
            done = {"contract_sha256": "hash", "variant": "graph", "evaluation_questions_used": 0,
                "tiny_weights_reused": False, "train_questions": 2048, "validation_questions": 256,
                "epochs": 1, "best_sha256": sha_file(full / "best.pt"), "best_validation_ce": .2, "selected_epoch": 1}
            atomic_json(full / "COMPLETE.json", done)
            selected, weights, pending = selected_controllers(root, manifest, ["graph"], lambda p: json.loads(p.read_text()))
            self.assertFalse(pending); self.assertEqual(selected["graph"]["epoch"], 1)
            self.assertEqual(weights["graph"], {"fake": 1})
            atomic_json(full / "COMPLETE.json", {**done, "evaluation_questions_used": 1})
            with self.assertRaisesRegex(RuntimeError, "Invalid completed"):
                selected_controllers(root, manifest, ["graph"], lambda p: checkpoint)

    def test_shards_reject_wrong_ids_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.json"
            rows = [{"sample_id": "a", "prediction": "A"}]
            self.assertIsNone(read_shard(path, "hash", rows))
            atomic_json(path, {"contract_sha256": "hash", "rows": rows, "rows_sha256": digest(rows)})
            self.assertEqual(read_shard(path, "hash", rows), rows)
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                read_shard(path, "hash", [{"sample_id": "b"}])
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                read_shard(path, "different", rows)

    def test_actual_generation_path_ignores_gold_and_targets(self):
        from test_dynamic_kg_attention import controller_fixture
        from test_kg_structure_attention import tiny_model
        module = entry(); model = tiny_model(); controller, layout = controller_fixture()
        controller.attach(model)
        c = {"layouts": {"a": layout}, "tokenizer": SimpleNamespace(eos_token_id=2, convert_tokens_to_ids=lambda _: 2),
             "generation": {"max_new_tokens": 3}}
        row = {"sample_id": "a", "text_prompt_ids": [1, 3, 4, 5, 6], "answer": "A", "target_ids": [20]}
        expected = module.generate_batch(model, controller, [row], c)
        changed = {**row, "answer": "Z", "target_ids": [21, 22]}
        self.assertEqual(module.generate_batch(model, controller, [changed], c), expected)


if __name__ == "__main__": unittest.main()

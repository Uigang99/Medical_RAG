"""CPU tests of the real training loop, optimizer rollback and resume boundary."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_proposition_kg_attention import fixture, row_fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prop_training", ROOT / "scripts/train_rag2_proposition_kg_attention.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Progress:
    def __init__(self):
        self.names = ["graph train epoch1", "graph val epoch1", "save"]
        self.index, self.weights = 0, [None] * 3
    def stage(self, index, total, **kwargs):
        import time
        self.index = index; self.stage_started = time.monotonic()
    def update(self, *args): pass
    def complete(self, *args): pass
    def log(self, *args): pass


def rows():
    records = []
    for i in range(6):
        r = row_fixture(i % 2)
        r.update(sample_id=str(i), dataset="medmcqa" if i % 2 == 0 else "medqa",
                 split="train" if i < 4 else "validation", choice_token_ids=[31, 32, 33, 34],
                 option_labels=["A", "B", "C", "D"], gold_index=i % 4)
        records.append(r)
    return records


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_all_safety_forwards_exclude_cudnn_and_restore_backend(self):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        wrapper = fixture(True)
        args = SimpleNamespace(seed=42, bias_penalty=.001)
        original = torch.nn.functional.scaled_dot_product_attention
        calls = []

        def checked_attention(*args, **kwargs):
            self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled(),
                             "Safety forward escaped the non-cuDNN backend context")
            calls.append(1)
            return original(*args, **kwargs)

        # Backend flags are inspectable on CPU: reproduce the escaped context
        # even without a CUDA device or a cuDNN execution plan.
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]):
            with patch.object(runner, "make_wrapper", return_value=wrapper), \
                 patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=checked_attention):
                report = runner.runtime_checks(args, wrapper.llm, rows(), 0)
            self.assertTrue(torch.backends.cuda.cudnn_sdp_enabled())
        self.assertGreater(len(calls), 0)
        self.assertGreater(report["controller_gradient_l1"], 0.)
        self.assertTrue(report["answer_suffix_and_gold_do_not_change_bias"])

    def test_choice_loss_gold_is_not_prompt_and_backpropagates(self):
        wrapper = fixture(True)
        data = rows()[:2]
        before = copy.deepcopy(data)
        ce, penalty, logits, result = runner.choice_losses(wrapper, data, 0)
        self.assertEqual(tuple(logits.shape), (2, 4))
        (ce + .001 * penalty).mean().backward()
        self.assertGreater(wrapper.controller.output.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(data, before)
        self.assertTrue(all(p.grad is None for p in wrapper.llm.parameters()))

    def test_oom_rolls_back_all_partial_gradients(self):
        plain, retry = fixture(True), fixture(True)
        opt_plain = torch.optim.AdamW(plain.controller.parameters(), lr=.001)
        opt_retry = torch.optim.AdamW(retry.controller.parameters(), lr=.001)
        data = rows()[:4]
        runner.train_window(plain, opt_plain, data, 0, runner.AdaptiveBatch(1, 1), .001)
        original = runner.backward_micro
        calls = [0]
        def failure(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2: raise torch.cuda.OutOfMemoryError("simulated CUDA out of memory")
            return original(*args, **kwargs)
        adaptive = runner.AdaptiveBatch(2, 2)
        with patch.object(runner, "backward_micro", side_effect=failure):
            result = runner.train_window(retry, opt_retry, data, 0, adaptive, .001)
        self.assertEqual(result["micro_batch"], 1)
        self.assertGreater(next(iter(adaptive.buckets.values()))["cooldown"], 0)
        for p, q in zip(plain.controller.parameters(), retry.controller.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-4)

    def test_optimizer_failure_is_not_retried(self):
        wrapper = fixture(True)
        optimizer = torch.optim.AdamW(wrapper.controller.parameters(), lr=.001)
        with patch.object(optimizer, "step", side_effect=torch.cuda.OutOfMemoryError("optimizer OOM")) as step:
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                runner.train_window(wrapper, optimizer, rows()[:2], 0, runner.AdaptiveBatch(2, 2), .001)
            self.assertEqual(step.call_count, 1)

    def test_length_order_keeps_every_example_and_is_deterministic(self):
        data = rows() * 3
        first = runner.epoch_order(data, 0, 42, 4)
        self.assertEqual(sorted(first), list(range(len(data))))
        self.assertEqual(first, runner.epoch_order(data, 0, 42, 4))
        self.assertNotEqual(first, runner.epoch_order(data, 1, 42, 4))

    def test_free_disk_does_not_change_contract(self):
        a = runner.arguments([])
        inputs = {"fixed": "data", "source_disk_free_bytes": 100}
        first = runner.contract_for(a, inputs)
        inputs["source_disk_free_bytes"] = 50
        self.assertEqual(first, runner.contract_for(a, inputs))

    def test_prepared_reuse_is_read_only_and_keeps_exact_split_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = runner.arguments(["--output-root", str(root / "new"),
                                  "--reuse-prepared-from", str(root / "old")])
            old = runner.contract_for(a, {"source": "fixed"})
            current = copy.deepcopy(old)
            current["code"]["scripts/train_rag2_proposition_kg_attention.py"] = "updated-trainer"
            current["code"]["medrag/generation/proposition_kg_attention.py"] = "updated-runtime"
            data = [{"split": "train"}] * 9000 + [{"split": "validation"}] * 1000
            donor = a.reuse_prepared_from
            runner.atomic_json(donor / "manifest.json", {"contract": old, "contract_sha256": runner.digest(old)})
            rows_path = donor / "prepared/rows.jsonl"
            runner.atomic_jsonl(rows_path, data)
            runner.atomic_json(donor / "prepared/COMPLETE.json", {
                "contract_sha256": runner.digest(old), "rows_sha256": runner.sha_file(rows_path)})
            before = {path: path.read_bytes() for path in donor.rglob("*") if path.is_file()}
            selected = runner.prepared_cache_source(a, current)
            self.assertEqual(selected, rows_path)
            tokenizer = object()
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch.object(runner, "build_rows", side_effect=AssertionError("must reuse cache")):
                result, loaded = runner.prepare(a, Progress(), runner.digest(current), selected)
                self.assertEqual(result, data)
                self.assertIs(loaded, tokenizer)
                with patch.object(runner, "jsonl", return_value=iter(data[:-1])):
                    with self.assertRaisesRegex(RuntimeError, "exact official train9000/val1000"):
                        runner.prepare(a, Progress(), runner.digest(current), selected)
            self.assertFalse(a.output_root.exists())
            self.assertEqual(before, {path: path.read_bytes() for path in donor.rglob("*") if path.is_file()})

    def test_prepared_reuse_rejects_contract_and_integrity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = runner.arguments(["--output-root", str(root / "new"),
                                  "--reuse-prepared-from", str(root / "old")])
            contract = runner.contract_for(a, {"source": "fixed"})
            donor = a.reuse_prepared_from
            manifest = {"contract": contract, "contract_sha256": runner.digest(contract)}
            runner.atomic_json(donor / "manifest.json", manifest)
            rows_path = donor / "prepared/rows.jsonl"
            runner.atomic_jsonl(rows_path, rows())
            meta = {"contract_sha256": runner.digest(contract), "rows_sha256": runner.sha_file(rows_path)}
            complete = donor / "prepared/COMPLETE.json"
            runner.atomic_json(complete, meta)
            for field in ("inputs", "model", "environment", "training"):
                current = copy.deepcopy(contract)
                current[field] = {"changed": True}
                with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "donor contract mismatch"):
                    runner.prepared_cache_source(a, current)
            for name in set(runner.CODE) - {"scripts/train_rag2_proposition_kg_attention.py",
                                             "medrag/generation/proposition_kg_attention.py"}:
                current = copy.deepcopy(contract)
                current["code"][name] = "changed"
                with self.subTest(code=name), self.assertRaisesRegex(RuntimeError, "donor contract mismatch"):
                    runner.prepared_cache_source(a, current)
            runner.atomic_json(donor / "manifest.json", dict(manifest, contract_sha256="bad"))
            with self.assertRaisesRegex(RuntimeError, "manifest digest mismatch"):
                runner.prepared_cache_source(a, contract)
            runner.atomic_json(donor / "manifest.json", manifest)
            for field in ("contract_sha256", "rows_sha256"):
                runner.atomic_json(complete, dict(meta, **{field: "bad"}))
                with self.subTest(integrity=field), self.assertRaisesRegex(RuntimeError, "cache mismatch"):
                    runner.prepared_cache_source(a, contract)

    def test_prepared_destination_cache_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = runner.arguments(["--output-root", str(root / "new"),
                                  "--reuse-prepared-from", str(root / "missing-donor")])
            contract = runner.contract_for(a, {"source": "fixed"})
            rows_path = a.output_root / "prepared/rows.jsonl"
            runner.atomic_jsonl(rows_path, rows())
            runner.atomic_json(a.output_root / "prepared/COMPLETE.json", {
                "contract_sha256": runner.digest(contract), "rows_sha256": runner.sha_file(rows_path)})
            self.assertEqual(runner.prepared_cache_source(a, contract), rows_path)

    def test_full_epoch_resume_after_train_before_validation(self):
        target = fixture().llm
        data = rows()
        baseline = {"macro2": 0., "answer_ce": 3., "elapsed_seconds": .1}
        with tempfile.TemporaryDirectory() as directory:
            a = SimpleNamespace(output_root=Path(directory), epochs=1, learning_rate=.001,
                                weight_decay=.01, effective_batch_size=2, max_micro_batch_size=2,
                                initial_micro_batch_size=1, memory_target=.88, seed=42,
                                checkpoint_steps=1, bias_penalty=.001, validation_batch_size=2)
            def factory(*args):
                return fixture(True)
            with patch.object(runner, "make_wrapper", side_effect=factory):
                with patch.object(runner, "evaluation", side_effect=RuntimeError("validation interrupted")):
                    with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                        runner.train_arm(a, target, data, 0, "graph", baseline, "contract", Progress(), 0)
                checkpoint = runner.load_torch(Path(directory) / "graph/last.pt")
                self.assertEqual(checkpoint["cursor"], 4)
                self.assertEqual(checkpoint["updates"], 2)
                outcome = runner.train_arm(a, target, data, 0, "graph", baseline, "contract", Progress(), 0)
                self.assertEqual(outcome["updates"], 2)
                self.assertEqual(outcome["test_questions_used"], 0)
                self.assertFalse(outcome["target_llm_weights_updated"])
                again = runner.train_arm(a, target, data, 0, "graph", baseline, "contract", Progress(), 0)
                self.assertEqual(outcome, again)


if __name__ == "__main__": unittest.main()

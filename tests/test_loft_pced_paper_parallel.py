import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_loft_pced_paper as core
from scripts import run_loft_pced_paper_parallel as parallel


class ParallelPcedTest(unittest.TestCase):
    def test_weighted_progress_bounds(self):
        empty = {dataset: (0, 0, 0) for dataset in parallel.DATASETS}
        full = {dataset: (100, 100, 100) for dataset in parallel.DATASETS}
        self.assertEqual(parallel.weighted_progress(empty), 0)
        self.assertEqual(parallel.weighted_progress(full), 10000)

    def test_gpu_gate_requires_physical_gpu1(self):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.mem_get_info", return_value=(50 * 2**30, 140 * 2**30)):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                core.gpu_gate(40)
            for gpu in ("0", "0,1"):
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": gpu}):
                    with self.assertRaisesRegex(RuntimeError, "GPU-1 only"):
                        core.gpu_gate(40)

    def test_workers_launch_on_gpu1_only(self):
        self.assertEqual(parallel.GPU, 1)
        self.assertEqual(len(parallel.WORKERS), 2)
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(parallel.subprocess, "Popen") as popen:
            for worker, datasets in parallel.WORKERS.items():
                _, log, _ = parallel.launch(Path(temp), worker, datasets[0])
                log.close()
            for call in popen.call_args_list:
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
                self.assertEqual(call.args[0][2], "all")

    def test_gpu_preflight_checks_gpu1_only(self):
        with mock.patch.object(parallel.subprocess, "check_output", return_value="100000\n") as check:
            self.assertEqual(parallel.gpu_preflight(), 100000)
            self.assertEqual(check.call_args.args[0][1:3], ["-i", "1"])
        with mock.patch.object(parallel.subprocess, "check_output", return_value="50000\n"):
            with self.assertRaisesRegex(RuntimeError, "GPU1"):
                parallel.gpu_preflight()

    def test_pilot_parity_rejects_changed_token(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, full = root / "reference", root / "full"
            for base in (reference, full):
                for kind in ("rankings", "predictions"):
                    (base / "hotpotqa" / kind).mkdir(parents=True)
                core.atomic_json(core.result_file(base / "hotpotqa/rankings", "q"),
                                 {"top90": [{"pid": "p"}]})
                core.atomic_json(core.result_file(base / "hotpotqa/predictions", "q"),
                                 {"generated_token_ids": [1], "expert_winners": [0]})
            with mock.patch.object(parallel, "PILOT_ROOT", reference):
                ranked, answered = set(), set()
                parallel.verify_pilot_pair(full, {"hotpotqa": ["q"]}, ranked, answered)
                self.assertEqual((len(ranked), len(answered)), (1, 1))
                core.atomic_json(core.result_file(full / "hotpotqa/predictions", "q"),
                                 {"generated_token_ids": [2], "expert_winners": [0]})
                with self.assertRaisesRegex(RuntimeError, "tokens/routing changed"):
                    parallel.verify_pilot_pair(full, {"hotpotqa": ["q"]}, ranked, set())


if __name__ == "__main__":
    unittest.main()

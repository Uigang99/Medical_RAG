import unittest
from types import SimpleNamespace

from scripts import run_medical_pced_qwen35 as qwen


class QwenPCEDAdapterTest(unittest.TestCase):
    def test_maps_logits_kwarg_without_altering_other_inputs(self):
        class FakeModel:
            model = object()
            device = "cpu"
            generation_config = SimpleNamespace(eos_token_id=5)

            def __call__(self, *args, **kwargs):
                return args, kwargs

        adapter = qwen.QwenPCEDAdapter(FakeModel())
        args, kwargs = adapter("tokens", num_logits_to_keep=1, use_cache=True)
        self.assertEqual(("tokens",), args)
        self.assertEqual({"logits_to_keep": 1, "use_cache": True}, kwargs)
        self.assertEqual("cpu", adapter.device)

    def test_qwen_run_preserves_reference_gamma_and_arms(self):
        self.assertEqual(("pced", "all-ranked", "no-rag", "single", "all"), qwen.ARMS)
        self.assertEqual("medical_pced_qwen35_9b_seeded100_v1", qwen.VERSION)


if __name__ == "__main__":
    unittest.main()

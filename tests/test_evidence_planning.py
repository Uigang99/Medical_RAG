import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from medrag import evidence_planning as task
import pilot_rag2_evidence_planning as run


class PlanningTests(unittest.TestCase):
    def sample(self):
        return {"sample_id": "q1", "dataset": "medqa", "row_idx": 0, "question": "Known fact. What next?",
                "options": {"A": "CHOICE_SENTINEL"}, "gold_answer": "GOLD_SENTINEL", "no_rag_answer": "PRED_SENTINEL",
                "no_rag_correct": True,
                "documents": [{"text": "Source fact.", "rank": i, "document_id": f"d{i}"} for i in range(1, 9)]}

    def plan(self):
        return {"answer_goal": "Identify the next action", "question_limitation": "", "steps": [
            {"id": 1, "depends_on": [], "action": "Check the applicable decision rule", "required_information": ["Decision rule"],
             "available_information": [{"fact": "Known fact", "source": "question", "quote": "Known fact."}],
             "additional_information_needed": ["Applicable rule"]}]}

    def test_inputs_allowlist_and_experts(self):
        requests = task.make_requests([self.sample()])
        self.assertEqual(len(requests), 9)
        for req in requests:
            self.assertNotIn("SENTINEL", req["prompt"])
        self.assertIsNone(requests[0]["document"])
        for i in range(1, 9):
            self.assertEqual(requests[i]["document"]["document_id"], f"d{i}")

    def test_open_ended_planning_policy(self):
        self.assertIn("plan must work for an open-ended question", task.SYSTEM)
        self.assertIn("Do not enumerate, compare, rank, or eliminate answer choices", task.SYSTEM)
        self.assertIn("do not invent hypothetical choices", task.SYSTEM)
        self.assertIn("Comparing quantities or conditions explicitly requested", task.SYSTEM)

    def test_sampling_ratios_and_determinism(self):
        population = []
        for ds, n, c in (("medmcqa", 4183, 2330), ("medqa", 1273, 807), ("mmlu_anatomy", 1089, 791)):
            population.extend({"sample_id": f"{ds}:{i}", "dataset": ds, "row_idx": i, "no_rag_correct": i < c} for i in range(n))
        cohort = task.select_cohort(population)
        self.assertEqual(len(cohort), 64)
        self.assertEqual(cohort, task.select_cohort(list(reversed(population))))
        self.assertEqual(sum(r["no_rag_correct"] for r in cohort), 38)
        self.assertEqual(sum(r["dataset"] == "medmcqa" for r in cohort), 41)

    def test_parser_not_semantic_grader(self):
        req = task.make_requests([self.sample()])[0]
        plan, flags = task.parse_plan(json.dumps(self.plan()), req, "stop")
        self.assertEqual(flags, [])
        self.assertIsNotNone(plan)
        bad = copy.deepcopy(self.plan())
        bad["steps"][0]["available_information"][0]["source"] = "document"
        bad["steps"][0]["depends_on"] = [2]
        _, flags = task.parse_plan(json.dumps(bad), req, "length")
        self.assertIn("quote_not_exactly_grounded", flags)
        self.assertIn("invalid_step_dependencies", flags)
        self.assertIn("output_truncated", flags)
        self.assertIsNone(task.parse_plan("{", req, "length")[0])

    def test_generation_resume_exact_path_without_gpu(self):
        requests = task.make_requests([self.sample()])[:2]
        text = json.dumps(self.plan())
        class FakeLLM:
            def generate(self, prompts, params, use_tqdm):
                return [SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1, 2], finish_reason="stop")]) for _ in prompts]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory))
            prompts = {r["request_id"]: r["prompt"] for r in requests}
            run.save_generation_batch(args, FakeLLM(), requests, prompts, None, "contract", 0)
            cached = run.load_cached(args.output_dir / "generations", requests, "contract")
            self.assertEqual(len(cached), 2)
            with self.assertRaises(RuntimeError):
                run.load_cached(args.output_dir / "generations", requests, "wrong")
            with self.assertRaises(RuntimeError):
                run.save_generation_batch(args, FakeLLM(), requests, prompts, None, "contract", 0)


if __name__ == "__main__":
    unittest.main()

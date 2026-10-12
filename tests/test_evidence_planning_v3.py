import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from medrag import evidence_planning_v3 as task
import pilot_rag2_evidence_planning as runner
import pilot_rag2_evidence_planning_v3 as revision
from tests import test_evidence_planning as original_tests


class PlanningV3Tests(unittest.TestCase):
    def plan(self):
        return {"answer_goal": "Determine the applicable action", "question_limitation": "", "steps": [
            {"id": 1, "operation": "MATCH", "action": "Check the supplied condition against the decision rule",
             "expected_output": "Whether the rule applies", "needs": [
                 {"information": "condition -- applicable rule", "question_quotes": ["Known fact."],
                  "document_quotes": [], "background_knowledge": "", "from_steps": [], "missing": "Decision rule"}]}]}

    def test_allowlist_and_new_contract(self):
        sample = original_tests.PlanningTests().sample()
        requests = task.make_requests([sample])
        self.assertEqual(len(requests), 9)
        for i, req in enumerate(requests):
            supplied = json.loads(req["prompt"].split("\n", 1)[1])
            self.assertEqual(set(supplied), {"QUESTION", "DOCUMENT"})
            self.assertNotIn("SENTINEL", req["prompt"])
            self.assertEqual(supplied["DOCUMENT"], "Source fact." if i else None)
        self.assertNotEqual(requests[0]["request_sha256"], runner.task.make_requests([sample])[0]["request_sha256"])

    def test_provenance_partial_support_and_unresolved_needs(self):
        req = task.make_requests([original_tests.PlanningTests().sample()])[1]
        plan = self.plan()
        self.assertEqual(task.parse_plan(json.dumps(plan), req, "stop")[1], [])
        need = plan["steps"][0]["needs"][0]
        need["document_quotes"] = ["Known fact."]
        self.assertIn("document_quotes_not_exactly_grounded", task.parse_plan(json.dumps(plan), req, "stop")[1])
        need.update(question_quotes=[], document_quotes=[], missing="")
        self.assertIn("need_without_support_or_missing", task.parse_plan(json.dumps(plan), req, "stop")[1])
        self.assertIn("output_truncated", task.parse_plan(json.dumps(plan), req, "length")[1])

    def test_dependencies_use_earlier_outputs(self):
        req = task.make_requests([original_tests.PlanningTests().sample()])[0]
        plan = self.plan()
        next_step = copy.deepcopy(plan["steps"][0])
        next_step.update(id=2, operation="COMBINE", expected_output="Integrated judgment")
        next_step["needs"][0].update(question_quotes=[], missing="", from_steps=[1])
        plan["steps"].append(next_step)
        self.assertEqual(task.parse_plan(json.dumps(plan), req, "stop")[1], [])
        next_step["needs"][0]["from_steps"] = [2]
        self.assertIn("invalid_step_dependencies", task.parse_plan(json.dumps(plan), req, "stop")[1])
        next_step["operation"] = "diagnose_medical_case"
        self.assertIsNone(task.parse_plan(json.dumps(plan), req, "stop")[0])

    def test_exact_runner_save_resume_reports(self):
        cohort = [original_tests.PlanningTests().sample()]
        requests = task.make_requests(cohort)
        text = json.dumps(self.plan())

        class FakeLLM:
            def generate(self, prompts, params, use_tqdm):
                return [SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1, 2], finish_reason="stop")]) for _ in prompts]

        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "task", task):
            args = SimpleNamespace(output_dir=Path(directory))
            prompts = {r["request_id"]: r["prompt"] for r in requests}
            runner.save_generation_batch(args, FakeLLM(), requests, prompts, None, "v3", 0)
            cached = runner.load_cached(args.output_dir / "generations", requests, "v3")
            runner.write_reports(args, cohort, requests, cached, {})
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["saved_plans"], 9)
            self.assertEqual(summary["parsed_schema_valid"], 9)
            self.assertEqual(len(list(runner.iter_jsonl(args.output_dir / "by_question.jsonl"))), 1)
            with self.assertRaises(RuntimeError):
                runner.load_cached(args.output_dir / "generations", requests, "v2")

    def test_refuses_old_output_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), previous_run=Path(directory))
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                revision.load_inputs(args)


if __name__ == "__main__":
    unittest.main()

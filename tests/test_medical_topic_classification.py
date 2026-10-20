import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from medrag import medical_topic_classification as m
from medrag.training.graph_input_data import atomic_jsonl


class TopicTests(unittest.TestCase):
    def row(self, name="doc"):
        return m.document({"stable_id": name, "source": "pmc", "restored_text": "A medical passage."},
                          "restored_text")

    def response(self, body, tokens=1000):
        return {"model": m.MODEL, "answers": {
            k: {"type": "noul", "noul": .6} for k in body["questions"]},
                "usage": {"input_tokens": tokens, "output_tokens": 100}}

    def test_one_call_all_topics_no_gold(self):
        body = m.payload(self.row())
        self.assertEqual(len(body["questions"]), 21)
        self.assertEqual(set(body["state"]), {"title", "passage"})
        client = Mock()
        client.post.return_value.status_code = 200
        client.post.return_value.json.return_value = self.response(body)
        result, _ = m.send(client, "mock-key", body)
        client.post.assert_called_once()
        self.assertEqual(len(m.validate_response(result, body["questions"])[0]), 21)

    def test_resume_dedup_add_category_budget_and_uncertain(self):
        with tempfile.TemporaryDirectory() as td:
            cache = m.Cache(Path(td) / "cache.sqlite3")
            try:
                row = self.row()
                subset = list(m.TOPICS)[:20]
                body = m.payload(row, subset)
                with self.assertRaisesRegex(RuntimeError, "budget"):
                    cache.reserve(body, 1e-10)
                attempt = cache.reserve(body, 1)
                with self.assertRaisesRegex(RuntimeError, "billed"):
                    cache.reserve(body, 1)
                cache.finish(attempt, body, self.response(body), .1)
                self.assertEqual(len(cache.scores(self.row("different-doc-id"))), 20)
                self.assertAlmostEqual(cache.spent, 1000 * .042 / 1e6)
                missing = list(set(m.TOPICS) - set(cache.scores(row)))
                self.assertEqual(len(missing), 1)
                new_body = m.payload(row, missing)
                attempt = cache.reserve(new_body, 1)
                cache.finish(attempt, new_body, self.response(new_body), .1)
                self.assertEqual(m.estimate([row], cache)["pending_requests"], 0)
                self.assertEqual(len(m.export_row(row, cache.scores(row))["scores"]), 21)
            finally:
                cache.close()
            cache = m.Cache(Path(td) / "cache.sqlite3")
            try:
                self.assertEqual(len(cache.scores(row)), 21)
            finally:
                cache.close()

    def test_no_silent_text_fallback_no_mutation(self):
        with self.assertRaises(ValueError):
            m.document({"stable_id": "a", "text": "wrong source"}, "restored_text")
        body = m.payload(self.row())
        data = self.response(body)
        data["answers"][next(iter(m.TOPICS))]["noul"] = True
        with self.assertRaises(ValueError):
            m.validate_response(data, body["questions"])
        out = m.export_row(self.row(), {k: .01 for k in m.TOPICS})
        self.assertEqual(out["status"], "unreviewed")
        self.assertNotIn("accepted_labels", out)

    def test_sampling_and_conflicting_id(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "docs.jsonl"
            rows = [{"stable_id": str(i), "source": s, "restored_text": f"Passage {i}"}
                    for i, s in enumerate(["pmc", "cpg"] * 4)]
            atomic_jsonl(p, rows)
            selected = m.select_documents([p], "restored_text", 4, 1)
            self.assertEqual([x["source"] for x in selected], ["cpg", "pmc"] * 2)
            self.assertEqual(len(m.select_documents([p, p], "restored_text", 0, 1)), 8)
            rows.append({**rows[0], "restored_text": "changed"})
            atomic_jsonl(p, rows)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                m.select_documents([p], "restored_text", 0, 1)

    def test_exact_entrypoint_mock_api_export_and_resume(self):
        script = Path(__file__).resolve().parents[1] / "scripts/classify_rag2_documents_jev.py"
        spec = importlib.util.spec_from_file_location("classify_jev_test", script)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "docs.jsonl"
            atomic_jsonl(source, [{"stable_id": str(i), "source": "pmc",
                                  "restored_text": "Exactly the same document."} for i in range(3)])
            args = ["--mode", "run", "--source", str(source), "--limit", "0",
                    "--output-root", str(root / "output"), "--cache-root", str(root / "cache")]
            with patch.object(m, "api_key", return_value="mock"), \
                    patch.object(m, "send", side_effect=lambda client, key, body: (self.response(body), .1)) as send:
                entry.main(args)
                entry.main(args)
                self.assertEqual(send.call_count, 1)
            output = [json.loads(s) for s in (root / "output/labels.jsonl").read_text().splitlines()]
            self.assertEqual(len(output), 3)
            self.assertTrue(all(len(x["scores"]) == 21 for x in output))
            self.assertFalse(json.loads((root / "output/summary.json").read_text())["quality_evaluated"])


if __name__ == "__main__":
    unittest.main()

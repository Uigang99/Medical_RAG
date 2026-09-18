import importlib.util
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rag2_statement_graph_binary_relation_gate.py"
spec = importlib.util.spec_from_file_location("binary_relation_gate", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class BinaryRelationGateTest(unittest.TestCase):
    def test_source_stratified_subset_is_balanced_and_deterministic(self):
        rows = [
            {"source": source, "proposition_id": f"{source}-{index}"}
            for source in ("cpg", "pmc", "pubmed", "textbooks")
            for index in range(10)
        ]
        first = module._source_stratified_subset(rows, 16, "development")
        second = module._source_stratified_subset(rows, 16, "development")
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["source"] for row in first),
            Counter({source: 4 for source in ("cpg", "pmc", "pubmed", "textbooks")}),
        )

    def test_parser_requires_one_valid_code_per_pair(self):
        row = {
            "candidate_pairs": [{
                "pair_id": "X::PAIR001", "gold_relations": [],
                "left": {"surface": "A", "kind": "condition"},
                "right": {"surface": "B", "kind": "condition"},
            }]
        }
        value = module.validate_output(row, '{"d":[{"p":"P001","c":"N"}]}')
        self.assertTrue(value["binary_decisions"][0]["valid"])
        self.assertEqual(value["binary_decisions"][0]["choice"], "N")

    def test_pair_isolation_preserves_each_pair_once(self):
        row = {
            "proposition_id": "P0", "candidate_pairs": [
                {"pair_id": "P0::PAIR001"}, {"pair_id": "P0::PAIR002"}
            ]
        }
        values = module.isolate_candidate_pairs([row])
        self.assertEqual(len(values), 2)
        self.assertEqual([item["candidate_pairs"][0]["pair_id"] for item in values], ["P0::PAIR001", "P0::PAIR002"])

    def test_balanced_metrics_do_not_reward_majority_only(self):
        decisions = [
            {"gold_positive": True, "choice": "N", "valid": True},
            {"gold_positive": False, "choice": "N", "valid": True},
            {"gold_positive": False, "choice": "N", "valid": True},
        ]
        value = module.binary_metrics(decisions)
        self.assertEqual(value["positive_relation"]["recall"], 0.0)
        self.assertEqual(value["negative_specificity"], 1.0)
        self.assertEqual(value["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

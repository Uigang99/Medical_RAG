import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/extract_rag2_statement_graph_pair_classifier_v5.py"
spec = importlib.util.spec_from_file_location("pair_classifier_v5", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.path.insert(0, str(ROOT / "scripts"))
spec.loader.exec_module(module)


def mention(mid, surface, kind, start, end):
    return {"mention_id": mid, "surface": surface, "kind": kind, "start": start, "end": end, "umls_cui": None, "umls_name": None}


def example_row():
    claim = "Penicillin-allergic patients can receive erythromycin."
    return {
        "proposition_id": "P1", "source": "textbooks", "claim": claim,
        "mentions": [
            mention("N1", "Penicillin-allergic patients", "population", 0, 28),
            mention("N2", "erythromycin", "drug_or_chemical", 41, 53),
        ],
    }


class PairClassifierV5Test(unittest.TestCase):
    def test_pair_options_fix_endpoints_and_directions(self):
        pairs, dropped = module.construct_pairs(example_row(), 48)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(pairs), 1)
        self.assertIn("R2L:ADMINISTERED_TO", pairs[0]["allowed_choices"])
        self.assertNotIn("L2R:ADMINISTERED_TO", pairs[0]["allowed_choices"])

    def test_no_relation_is_valid_without_evidence(self):
        row = {**example_row()}
        row["candidate_pairs"], _ = module.construct_pairs(row, 48)
        raw = '{"decisions":[{"pair_id":"P1::PAIR001","choice":"NO_RELATION","evidence_spans":[],"negated":false,"uncertain":false,"modality":"asserted"}]}'
        value = module.validate_decisions(row, raw)
        decision = value["pair_decisions"][0]
        self.assertTrue(decision["classification_valid"])
        self.assertFalse(decision["positive"])

    def test_positive_relation_requires_exact_evidence(self):
        row = {**example_row()}
        row["candidate_pairs"], _ = module.construct_pairs(row, 48)
        raw = '{"decisions":[{"pair_id":"P1::PAIR001","choice":"R2L:ADMINISTERED_TO","evidence_spans":["not in source"],"negated":false,"uncertain":false,"modality":"can"}]}'
        value = module.validate_decisions(row, raw)
        self.assertFalse(value["pair_decisions"][0]["classification_valid"])

    def test_positive_relation_resolves_fixed_orientation(self):
        row = {**example_row()}
        row["candidate_pairs"], _ = module.construct_pairs(row, 48)
        raw = '{"decisions":[{"pair_id":"P1::PAIR001","choice":"R2L:ADMINISTERED_TO","evidence_spans":["Penicillin-allergic patients can receive erythromycin"],"negated":false,"uncertain":false,"modality":"can"}]}'
        value = module.validate_decisions(row, raw)
        decision = value["pair_decisions"][0]
        self.assertTrue(decision["classification_valid"])
        self.assertEqual(decision["subject"]["surface"], "erythromycin")
        self.assertEqual(decision["object"]["surface"], "Penicillin-allergic patients")

    def test_compact_response_resolves_local_pair_and_choice_codes(self):
        row = {**example_row()}
        row["candidate_pairs"], _ = module.construct_pairs(row, 48)
        raw = '{"d":[{"p":"P001","c":"R:ADMINISTERED_TO","e":["Penicillin-allergic patients can receive erythromycin"],"n":false,"u":false,"m":"can"}]}'
        decision = module.validate_decisions(row, raw)["pair_decisions"][0]
        self.assertTrue(decision["classification_valid"])
        self.assertEqual(decision["choice"], "R2L:ADMINISTERED_TO")

    def test_prompt_has_no_prior_relation_prediction(self):
        row = {**example_row(), "statements": [{"relation_type": "legacy wrong"}]}
        row["candidate_pairs"], _ = module.construct_pairs(row, 48)
        prompt = module.pair_prompt(row)
        self.assertNotIn("legacy wrong", prompt)
        self.assertIn('"choice_sets"', prompt)
        self.assertIn('"N"', prompt)


if __name__ == "__main__":
    unittest.main()

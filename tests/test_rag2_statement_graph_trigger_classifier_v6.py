import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/extract_rag2_statement_graph_trigger_classifier_v6.py"
spec = importlib.util.spec_from_file_location("trigger_classifier_v6", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.path.insert(0, str(ROOT / "scripts"))
spec.loader.exec_module(module)


def mention(mid, surface, kind, start, end, cui=None):
    return {
        "mention_id": mid,
        "surface": surface,
        "kind": kind,
        "start": start,
        "end": end,
        "umls_cui": cui,
        "umls_name": None,
        "umls_status": "linked_confident" if cui else "not_found",
        "umls_candidates": [],
        "globally_resolvable": bool(cui),
    }


def core(mid, surface, kind, start, end):
    return {
        **mention(mid, surface, kind, start, end),
        "core_id": mid,
    }


class TriggerClassifierV6Test(unittest.TestCase):
    def test_node_gate_rejects_verb_and_generic_population(self):
        verb = mention("N1", "decreased", "finding_or_outcome", 0, 9)
        patient = mention("N2", "patients", "population", 10, 18)
        self.assertFalse(module.mention_decision(verb, {"root_pos": "VERB"})["keep"])
        self.assertFalse(module.mention_decision(patient, {"root_pos": "NOUN"})["keep"])

    def test_node_gate_keeps_specific_noun(self):
        disease = mention("N1", "arterial hypertension", "condition", 0, 21)
        decision = module.mention_decision(disease, {"root_pos": "NOUN", "root_text": "hypertension"})
        self.assertTrue(decision["keep"])
        self.assertEqual(decision["resolved_kind"], "condition")

    def test_node_gate_rejects_prepositional_fragment(self):
        fragment = mention("N1", "after atrial fibrillation", "condition", 0, 25)
        decision = module.mention_decision(fragment, {"root_pos": "NOUN", "root_text": "fibrillation"})
        self.assertFalse(decision["keep"])

    def test_relation_candidate_needs_trigger_and_fixes_roles(self):
        claim = "Aspirin treats arterial hypertension."
        row = {
            "proposition_id": "P1",
            "claim": claim,
            "core_mentions": [
                core("C1", "Aspirin", "drug_or_chemical", 0, 7),
                core("C2", "arterial hypertension", "condition", 15, 36),
            ],
        }
        values, dropped = module.construct_relation_candidates(row, 32, 180)
        self.assertEqual(dropped, 0)
        treats = [value for value in values if value["relation_type"] == "TREATS"]
        self.assertEqual(len(treats), 1)
        self.assertEqual(treats[0]["subject"]["surface"], "Aspirin")
        self.assertEqual(treats[0]["object"]["surface"], "arterial hypertension")

    def test_relation_candidate_absent_without_explicit_trigger(self):
        claim = "Aspirin and arterial hypertension were evaluated."
        row = {
            "proposition_id": "P1",
            "claim": claim,
            "core_mentions": [
                core("C1", "Aspirin", "drug_or_chemical", 0, 7),
                core("C2", "arterial hypertension", "condition", 12, 33),
            ],
        }
        values, _ = module.construct_relation_candidates(row, 32, 180)
        self.assertEqual(values, [])

    def test_binary_verifier_cannot_change_endpoints_or_direction(self):
        claim = "Aspirin treats arterial hypertension."
        row = {
            "proposition_id": "P1",
            "source": "textbooks",
            "claim": claim,
            "mention_decisions": [],
            "core_mentions": [
                core("C1", "Aspirin", "drug_or_chemical", 0, 7),
                core("C2", "arterial hypertension", "condition", 15, 36),
            ],
        }
        row["relation_candidates"], _ = module.construct_relation_candidates(row, 32, 180)
        assertion_id = row["relation_candidates"][0]["assertion_id"]
        raw = '{"d":[{"i":"A001","j":"S","p":"asserted"}]}'
        value = module.validate_verification(row, raw)
        decision = value["verified_assertions"][0]
        self.assertTrue(decision["positive"])
        self.assertEqual(decision["assertion_id"], assertion_id)
        self.assertEqual(decision["subject"]["surface"], "Aspirin")

    def test_prompt_contains_no_prior_predictions(self):
        row = {
            "proposition_id": "P1",
            "claim": "Aspirin treats arterial hypertension.",
            "core_mentions": [
                core("C1", "Aspirin", "drug_or_chemical", 0, 7),
                core("C2", "arterial hypertension", "condition", 15, 36),
            ],
            "legacy_relation": "WRONG",
        }
        row["relation_candidates"], _ = module.construct_relation_candidates(row, 32, 180)
        prompt = module.verifier_prompt(row)
        self.assertNotIn("WRONG", prompt)
        self.assertIn('"r":"TREATS"', prompt)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/build_rag2_context_preserving_proposition_graph_pilot.py"
SPEC=importlib.util.spec_from_file_location("context_graph",SCRIPT)
MODULE=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(MODULE)

class ContextPreservingGraphTests(unittest.TestCase):
    def test_patient_specific_fact_is_not_filtered(self):
        row={"claim":"The admitted woman's troponin was elevated.","candidate_hints":[],"gold_entities":[]}
        parsed=MODULE.parse_prediction(row,'{"e":["admitted woman","troponin","elevated"]}')
        self.assertTrue(parsed["valid_output"])
        self.assertEqual(len(parsed["predicted_entities"]),3)

    def test_generic_lexical_bridge_is_rejected(self):
        self.assertIsNone(MODULE.lexical_key("patients","population"))

    def test_specific_lexical_bridge_is_allowed(self):
        self.assertEqual(MODULE.lexical_key("platelet aggregation","finding_or_outcome"),"LEX::finding_or_outcome::platelet aggregation")

    def test_contained_raw_mention_is_not_used_for_normalization(self):
        entity={"surface":"patients with partial renal recovery","start":0,"end":36}
        mentions=[{"surface":"renal recovery","start":22,"end":36}]
        self.assertIsNone(MODULE.exact_raw_mention(entity,mentions))

if __name__=="__main__": unittest.main()

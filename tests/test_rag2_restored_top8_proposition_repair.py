import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from retry_rag2_restored_top8_gemma_propositions import reasons_for_retry, machine_valid


class RepairSelectionTest(unittest.TestCase):
    def record(self,**changes):
        data=dict(parse_status='parsed',finish_reason='stop',propositions=[{'evidence_exact_match':True}],parse_errors=[],source_status='boundary_expanded_candidate')
        data.update(changes)
        return data

    def test_flagged_failures_and_empty_are_selected_but_source_warning_alone_is_not(self):
        self.assertEqual(reasons_for_retry(self.record(parse_status='parse_failure',propositions=[],parse_errors=['invalid_json'])),['parse_failure'])
        self.assertEqual(reasons_for_retry(self.record(parse_status='empty',propositions=[],parse_errors=['empty_propositions'])),['empty'])
        self.assertEqual(reasons_for_retry(self.record(source_status='review')),[])
        self.assertEqual(reasons_for_retry(self.record(finish_reason='length',propositions=[{'evidence_exact_match':False}])),['truncated','nonexact_evidence'])

    def test_machine_valid_requires_complete_exact_nonempty_replacement(self):
        row=dict(parsed={'propositions':[{'evidence_exact_match':True}]},finish_reason='stop',parse_errors=[],original_propositions=1)
        self.assertTrue(machine_valid(row))
        self.assertFalse(machine_valid(dict(row,original_propositions=2)))
        self.assertFalse(machine_valid(dict(row,finish_reason='length')))
        self.assertFalse(machine_valid(dict(row,parsed={'propositions':[]})))
        self.assertFalse(machine_valid(dict(row,parsed={'propositions':[{'evidence_exact_match':False}]})))


if __name__=='__main__':unittest.main()

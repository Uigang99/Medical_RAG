from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import pilot_rag2_kev_information_roles as pilot


class InformationRoleTests(unittest.TestCase):
    def test_independent_roles_and_no_gold_inputs(self):
        spec = json.loads((pilot.ROOT / 'docs/kev_information_roles_v2.json').read_text())
        row = {'id': 'x', 'text': 'X causes Y.', 'gold': 'SECRET_ANSWER'}
        job = pilot.jobs_for([row], spec)[0]
        self.assertNotIn('SECRET', json.dumps(job['body']))
        self.assertEqual(set(job['body']['questions']), set(pilot.ROLES))
        for q in job['body']['questions'].values():
            self.assertEqual(set(q['criteria']), {'YES', 'NO', 'UNCERTAIN'})

    def test_multiple_yes_and_optional_reference(self):
        answers = {r: {'choice': 'NO'} for r in pilot.ROLES}
        answers['METHOD']['choice'] = answers['NORM']['choice'] = 'YES'
        answers['EXPLAIN']['choice'] = 'UNCERTAIN'
        refs = {'x': {'yes': ['METHOD', 'NORM'], 'optional': ['EXPLAIN']}}
        result = pilot.metrics([{'id': 'x', 'text': 'Do X to prevent Y.', 'answers': answers}], refs)
        self.assertEqual(result['micro_precision'], 1)
        self.assertEqual(result['micro_recall'], 1)
        self.assertEqual(result['exact_on_unambiguous_n'], 1)
        self.assertEqual(result['by_role']['EXPLAIN']['ambiguous_excluded'], 1)

    def test_uncertain_is_not_scored_as_correct_negative(self):
        answers = {r: {'choice': 'NO'} for r in pilot.ROLES}
        answers['DEFINE']['choice'] = 'UNCERTAIN'
        result = pilot.metrics([{'id': 'x', 'text': 'X.', 'answers': answers}], {'x': {'yes': []}})
        self.assertEqual(result['exact_on_unambiguous_n'], 0)
        self.assertEqual(result['by_role']['DEFINE']['tn'], 0)
        self.assertEqual(result['uncertain_n'], 1)

    def test_deduplication_and_cross_split_rejection(self):
        def prop(text, doc):
            return {'claim': text, 'document_id': doc, 'proposition_id': doc + text,
                    'source': 'textbooks'}
        data = {'version': 'kev_abc_train_input_pilot_v1', 'pairs': [
            {'case_id': 'T001', 'split': 'development', 'P1': prop('X.', 'd1'), 'P2': prop(' x. ', 'd1')},
            {'case_id': 'T002', 'split': 'confirmation', 'P1': prop('Y.', 'd2'), 'P2': prop('Z.', 'd2')}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cohort.json'
            path.write_text(json.dumps(data))
            rows = pilot.cohort(path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['aliases'], ['T001P2'])
            data['pairs'][1]['P1']['claim'] = 'X.'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'crosses the split'):
                pilot.cohort(path)
            data['pairs'][1]['P1'] = prop('Y.', 'd1')
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'documents cross'):
                pilot.cohort(path)


if __name__ == '__main__':
    unittest.main()

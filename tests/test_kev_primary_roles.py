from __future__ import annotations

import json
import unittest
from scripts import pilot_rag2_kev_primary_roles as p


class PrimaryRoleTests(unittest.TestCase):
    def test_selected_prompt_schema(self):
        spec=json.loads((p.ROOT/'docs/kev_primary_roles_p3.json').read_text())
        self.assertEqual(spec['version'],'p3')
        self.assertEqual(set(spec['primary_criteria']),set(p.ROLES)|{'OTHER','UNCERTAIN'})
        self.assertEqual(set(spec['secondary_roles']),set(p.ROLES))
        self.assertEqual(set(spec['secondary_criteria']),{'YES','NO','UNCERTAIN'})

    def spec(self):
        return {'primary_instruction':'Choose the central role.',
                'primary_criteria':{r:r for r in p.ROLES},
                'secondary_instruction':'Choose only additional roles.',
                'secondary_roles':{r:r for r in p.ROLES},
                'secondary_criteria':{'YES':'present','NO':'absent','UNCERTAIN':'unclear'}}

    def test_direct_choice_and_secondary_exclusion(self):
        row={'id':'x','text':'X causes Y.','answer':'SECRET'}
        main=p.make_job(row,self.spec())
        self.assertEqual(list(main['body']['questions']),['primary'])
        self.assertEqual(main['body']['state'],{'proposition':row['text']})
        sub=p.make_job(row,self.spec(),'EXPLAIN')
        self.assertNotIn('EXPLAIN',sub['body']['questions'])
        self.assertEqual(len(sub['body']['questions']),5)
        self.assertNotEqual(main['id'],sub['id'])
        self.assertNotEqual(sub['id'],p.make_job(row,self.spec(),'DESCRIBE')['id'])

    def test_auxiliary_error_does_not_invalidate_core(self):
        row={'id':'x','primary':{'choice':'EXPLAIN'},
             'secondary':{'METHOD':{'choice':'YES'},'DESCRIBE':{'choice':'NO'}}}
        refs={'x':{'primary':'EXPLAIN','acceptable_primary':['EXPLAIN'],
                   'allowed_secondary':[],'essential_secondary':[]}}
        m=p.evaluate([row],refs)
        self.assertEqual(m['primary_exact_n'],1)
        self.assertEqual(m['secondary_precision'],0)
        self.assertEqual(m['core_anywhere_n'],1)

    def test_acceptable_alternative_separate_from_preferred(self):
        row={'id':'x','primary':{'choice':'METHOD'},'secondary':{'NORM':{'choice':'YES'}}}
        refs={'x':{'primary':'NORM','acceptable_primary':['NORM','METHOD'],
                   'allowed_secondary':[],'essential_secondary':['NORM']}}
        m=p.evaluate([row],refs)
        self.assertEqual(m['primary_exact_n'],0)
        self.assertEqual(m['primary_acceptable_n'],1)
        self.assertEqual(m['essential_secondary_recovered'],1)

    def test_core_in_secondary_is_not_primary_success(self):
        row={'id':'x','primary':{'choice':'DESCRIBE'},'secondary':{'EXPLAIN':{'choice':'YES'}}}
        refs={'x':{'primary':'EXPLAIN','acceptable_primary':['EXPLAIN'],
                   'allowed_secondary':['DESCRIBE'],'essential_secondary':[]}}
        m=p.evaluate([row],refs)
        self.assertEqual(m['primary_acceptable_n'],0)
        self.assertEqual(m['core_anywhere_n'],1)


if __name__=='__main__':unittest.main()

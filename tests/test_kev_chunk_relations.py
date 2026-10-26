import unittest
from types import SimpleNamespace
from medrag import kev_chunk_relations as s


class ChunkRelationsTest(unittest.TestCase):
    def test_local_checkpoint_paths_preserve_full_weights(self):
        from medrag.evidence_graph_runtime import localize_checkpoint
        full=SimpleNamespace(full=True,path='/tmp/kev27',meta=SimpleNamespace(base='Qwen/Qwen3.8-27B',base_revision='rev'))
        localize_checkpoint(full,'/tmp/unrelated-base')
        self.assertEqual(full.meta.base,'/tmp/kev27')
        self.assertIsNone(full.meta.base_revision)
        adapter=SimpleNamespace(full=False,meta=SimpleNamespace(base='Qwen/Qwen3.5-9B-Base',base_revision='rev'))
        localize_checkpoint(adapter,'/tmp/base9')
        self.assertEqual(adapter.meta.base,'/tmp/base9')
        invalid=SimpleNamespace(full=True,meta=SimpleNamespace(base='unknown'))
        with self.assertRaises(RuntimeError):localize_checkpoint(invalid,'/tmp/base')

    def test_independent_directions_and_no_question(self):
        pair = {"pair_id": "pair", "A": {"text": "literal A"}, "B": {"text": "literal B"}}
        jobs = [s.make_job(pair, task) for task in s.TASKS]
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len({j['id'] for j in jobs}), 6)
        self.assertEqual(sum(j['direction'] is not None for j in jobs), 4)
        for job in jobs:
            self.assertEqual(set(job['body']['state']), {'rules', 'chunk_A', 'chunk_B'})
            self.assertEqual(len(job['body']['questions']), 1)
            self.assertEqual(set(job['body']['questions']['RELATION']['criteria']), {'YES','NO','UNCERTAIN'})

    def test_only_yes_edges_and_reverse_direction(self):
        pair = {'pair_id':'p', 'case_id':'x', 'A':{'id':'a','text':'a'}, 'B':{'id':'b','text':'b'}}
        cohort={'pairs':[pair], 'questions':[{'case_id':'Q','sample_id':'q','chunks':[pair['A'],pair['B']],'pairs':['p']}]}
        jobs=[s.make_job(pair,t) for t in s.TASKS]
        results={j['id']:{'answers':{'RELATION':{'choice':'YES' if j['task']=='EXPLAINS_BA' else 'UNCERTAIN',
                                             'probabilities':{'YES':.8,'NO':.1,'UNCERTAIN':.1}}}} for j in jobs}
        rows,graphs,counts=s.materialize(cohort,jobs,results)
        self.assertEqual(len(rows),1)
        self.assertEqual(graphs[0]['graphs']['A'],[])
        self.assertEqual(graphs[0]['graphs']['C'],[])
        edge=graphs[0]['graphs']['B'][0]
        self.assertEqual((edge['source'],edge['target']),('b','a'))
        self.assertIsNone(edge['supporting_spans'])

    def test_refined_prompt_keeps_inputs_tasks_and_separate_cache(self):
        pair={'pair_id':'p','A':{'text':'alpha'},'B':{'text':'beta'}}
        for task in s.TASKS:
            old=s.make_job(pair,task);new=s.make_job(pair,task,'v2')
            self.assertNotEqual(old['id'],new['id'])
            self.assertEqual(old['direction'],new['direction'])
            self.assertEqual(new['body']['state']['chunk_A'],'alpha')
            self.assertEqual(new['body']['state']['chunk_B'],'beta')
            self.assertEqual(set(new['body']['questions']['RELATION']['criteria']),set(s.CRITERIA))
        with self.assertRaises(ValueError):s.make_job(pair,'OVERLAP','unknown')


if __name__ == '__main__': unittest.main()

import unittest
from medrag import kev_chunk_relations as s


class ChunkRelationsTest(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()

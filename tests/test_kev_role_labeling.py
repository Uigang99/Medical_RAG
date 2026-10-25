from __future__ import annotations

from argparse import Namespace
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from medrag.kev_role_labeling import RoleStore, output_parity, pack
from scripts import label_rag2_kev_roles_full as full


def spec():
    return json.loads((full.ROOT/'docs/kev_primary_roles_p3.json').read_text())


def answers(job):
    result = {}
    for key, q in job['body']['questions'].items():
        winner = 'DESCRIBE' if key == 'primary' else 'NO'
        result[key] = {'choice':winner,'confidence':1.,
                      'probabilities':{x:float(x==winner) for x in q['criteria']}}
    return result


class FakeJudge:
    loads = 0
    fail_secondary_once = False
    fresh_requests = 0
    def __init__(self, runtime, adapter, base, cache, batch_rows):
        FakeJudge.loads += 1
        self.cache, self.batch_rows = cache, batch_rows
        self.oom_splits, self.batch_disabled = 0, False

    def score(self, jobs, stage, emit):
        if self.fail_secondary_once and 'selected_primary_role' in jobs[0]['body']['state']:
            FakeJudge.fail_secondary_once = False
            raise RuntimeError('simulated interruption after primary cache')
        out = {}
        for job in jobs:
            value = self.cache.get(job)
            if value is None:
                value = self.cache.put(job,answers(job))
                FakeJudge.fresh_requests += 1
            out[job['id']] = value
        return out


class SilentProgress:
    def __init__(self, names):
        self.index = 0
    def stage(self, index, *a, **kw):
        self.index = index
    def update(self, *a, **kw): pass
    def complete(self, *a, **kw): pass
    def log(self, *a, **kw): pass
    def close(self): pass


class RoleLabelingTests(unittest.TestCase):
    def test_semantic_cache_reuse_preserves_proposition_ids(self):
        j = full.make_job({'id':'p1','text':'One observation.'},spec())
        k = full.make_job({'id':'p2','text':'One observation.'},spec())
        self.assertEqual(j['id'],k['id'])
        with tempfile.TemporaryDirectory() as root:
            s = RoleStore(root,'h')
            s.put(j,answers(j)); s.close()
            s = RoleStore(root,'h')
            self.assertEqual(s.get(k)['answers'],answers(j))
            self.assertEqual(s.count(),0)  # primary-only cache is NOT a completed label
            s.save_labels([{'index':0,'proposition_id':'p1'}, {'index':1,'proposition_id':'p2'}])
            self.assertEqual(s.completed(0,1),{0,1})
            s.close()
            with self.assertRaises(RuntimeError): RoleStore(root,'other')

    def test_corrupt_decision_rejected(self):
        j = full.make_job({'id':'p','text':'Some evidence.'},spec())
        bad = answers(j)
        bad['primary']['choice'] = 'NORM'
        with tempfile.TemporaryDirectory() as root:
            s = RoleStore(root,'h')
            with self.assertRaises(ValueError): s.put(j,bad)
            self.assertIsNone(s.get(j)); s.close()

    def test_secondary_preserves_all_remaining_roles(self):
        row = {'id':'p','text':'Some evidence.','gold':'SECRET'}
        j = full.make_job(row,spec(),'DESCRIBE')
        self.assertEqual(len(j['body']['questions']),5)
        self.assertNotIn('gold',str(j))
        self.assertEqual(len(full.make_job(row,spec(),'OTHER')['body']['questions']),6)

    def test_batch_rows_not_request_count(self):
        items = [('a',{},[0]*5),('b',{},[0]*5),('c',{},[0]*6)]
        self.assertEqual([len(x) for x in pack(items,10)],[2,1])

    def test_parity_requires_winner_and_probability_agreement(self):
        self.assertTrue(output_parity([[[.8,.2]]],[[[.801,.199]]]))
        self.assertFalse(output_parity([[[.501,.499]]],[[[.499,.501]]]))
        self.assertFalse(output_parity([[[.8,.2]]],[[[.95,.05]]]))

    def test_small_pipeline_partial_and_completed_resume_without_gpu(self):
        # Exercise the real orchestration/export with fake KEV only. Last row has
        # no pair/edge information at all, yet must still be labeled.
        import torch
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            source_path = root/'source.sqlite3'
            db = sqlite3.connect(source_path)
            db.execute('CREATE TABLE proposition(idx INTEGER,pid TEXT,did TEXT,claim TEXT)')
            db.executemany('INSERT INTO proposition VALUES (?,?,?,?)',
                           [(0,'p1','d1','same'),(1,'p2','d2','same'),(2,'p3','d3','isolated')])
            db.commit(); db.close()
            def source(a):
                d = sqlite3.connect(source_path); d.row_factory=sqlite3.Row
                return d,{'empty_document_slots':0},{'test':'fixture'}
            a = Namespace(preflight=False,output_root=root/'out',chunk_size=2,
                          max_batch_rows=48,memory_fraction=.88)
            for name,value in {'N':3,'read_source':source,'preflight_budget':lambda *a:100,
                'contract':lambda *a:({'test':'contract'},spec(),root,root),
                'Progress':SilentProgress,'LocalDecisions':FakeJudge,
                'tune_batch':lambda *a:{'selected_rows':48}}.items():
                stack.enter_context(patch.object(full,name,value))
            stack.enter_context(patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}))
            stack.enter_context(patch.object(torch.cuda,'set_per_process_memory_fraction'))
            stack.enter_context(patch.object(torch,'manual_seed'))
            FakeJudge.loads=0
            FakeJudge.fresh_requests=0
            FakeJudge.fail_secondary_once=True
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                full.run(a)
            self.assertFalse((a.output_root/'labels.jsonl').exists())
            full.run(a)
            first = (a.output_root/'labels.jsonl').read_text()
            labels = [json.loads(x) for x in first.splitlines()]
            self.assertEqual([x['proposition_id'] for x in labels],['p1','p2','p3'])
            self.assertEqual([x['text'] for x in labels],['same','same','isolated'])
            self.assertTrue(all(len(x['secondary'])==5 for x in labels))
            full.run(a)
            self.assertEqual(FakeJudge.loads,2)
            self.assertEqual(FakeJudge.fresh_requests,4)  # two unique texts, two stages; no repeated primary
            self.assertEqual((a.output_root/'labels.jsonl').read_text(),first)
            self.assertEqual(json.loads((a.output_root/'summary.json').read_text())['missing_labels'],0)


if __name__ == '__main__':
    unittest.main()

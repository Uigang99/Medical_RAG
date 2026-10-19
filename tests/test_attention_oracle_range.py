"""Tiny CPU Llama tests through real scoring, backward, bounds and checkpoint resume."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from tests.test_attention_oracle import fixture, Progress as BaseProgress
from medrag.training.attention_oracle import score
from medrag.training.attention_oracle_range import check_seed, SCALES, BOUNDS, summarize
from scripts import pilot_rag2_attention_oracle_range as runner


class Progress(BaseProgress):
    def __init__(self):
        super().__init__(); self.names=['']*8


def seed_fixture():
    w,row=fixture(); zero=score(w,row,0)
    # Deliberately nonzero source table, so an accidental zero restart is observable.
    values=torch.tensor([[2.,-1.,.5],[-.5,2.,0.],[.1,.2,-2.]])
    w.controller.assign(values)
    seed=dict(sample_id=row['sample_id'],ordered_pairs=row['pairs'],values=values.tolist(),best=score(w,row,0))
    return w,row,zero,seed


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_seed_contract_and_no_hidden_clipping(self):
        w,row,zero,seed=seed_fixture()
        values=check_seed(row,seed)
        w.controller.assign(values*4)
        self.assertEqual(float(w.controller.values.detach().abs().max()),8.)
        for field,value in [('sample_id','other'),('ordered_pairs',list(reversed(row['pairs']))),
                            ('values',[[9.,0.,0.]]*3),('values',[[float('nan'),0.,0.]]*3)]:
            wrong=copy.deepcopy(seed); wrong[field]=value
            with self.assertRaises(RuntimeError):check_seed(row,wrong)
        with self.assertRaises(RuntimeError):runner.replay_check(zero,seed['best'],'changed seed')

    def test_scaling_independent_fresh_arms_and_json_summary(self):
        w,row,zero,seed=seed_fixture(); original=copy.deepcopy(seed)
        frozen={k:v.clone() for k,v in w.llm.state_dict().items()}
        i=row['sample_id']
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp),steps=4)
            progress=Progress()
            scales=runner.scale_only(a,w,[row],{i:seed},{i:zero},0,'h',progress)
            self.assertEqual(progress.updates,5)
            for scale in SCALES:
                w.controller.assign(check_seed(row,seed)*scale)
                self.assertEqual(scales[i]['scales'][f'{scale:g}'],score(w,row,0))
            optimized={i:{}}
            for bound in reversed(BOUNDS):  # Execution order must not affect initialization.
                r=runner.optimize_bound(a,w,row,seed,0,'h',bound,.1,progress)
                optimized[i][bound]=r
                self.assertEqual(r['initial_values'],seed['values'])
                self.assertEqual(r['milestones']['0']['best'],seed['best'])
                self.assertEqual(set(r['milestones']),{'0','2','4'})
                self.assertEqual(len(r['history']),4)
                self.assertLessEqual(max(abs(v) for x in r['endpoint_values'] for v in x),bound)
                self.assertLessEqual(r['best']['ce'],seed['best']['ce'])
                self.assertEqual(r['history'][0]['score_before_update'],seed['best'])
            report=summarize([row],{i:seed},scales,optimized,a.steps,'h')
            runner.atomic_json(Path(tmp)/'SUMMARY.json',report)
            self.assertEqual(report['missing'],0)
            self.assertEqual(report['conditions']['scale_1']['vs_saved_oracle']['ce_reduction'],0)
            self.assertEqual(len(report['paired_contrasts']),10)
            self.assertEqual(seed,original)
            for k,v in w.llm.state_dict().items(): torch.testing.assert_close(v,frozen[k],atol=0,rtol=0)
            self.assertTrue(all(p.grad is None for p in w.llm.parameters()))
            # Completed workflow resumes without further scoring or optimizer steps.
            p=Progress()
            self.assertEqual(runner.scale_only(a,w,[row],{i:seed},{i:zero},0,'h',p),scales)
            for bound in BOUNDS:
                self.assertEqual(runner.cached_steps(a,[row],bound,'h'),4)
                self.assertEqual(runner.optimize_bound(a,w,row,seed,0,'h',bound,.1,p),optimized[i][bound])
            self.assertEqual(p.updates,0)

    def test_partial_resume_and_completed_checkpoint_replay(self):
        w,row,_,seed=seed_fixture()
        class Interrupted(Progress):
            def update(self,n=1):
                super().update(n)
                if self.updates==7: raise RuntimeError('intentional interruption')
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp)/'resume',steps=8)
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                runner.optimize_bound(a,w,row,seed,0,'h',4.,.1,Interrupted())
            self.assertEqual(runner.cached_steps(a,[row],4.,'h'),5)
            p=Progress(); resumed=runner.optimize_bound(a,w,row,seed,0,'h',4.,.1,p)
            self.assertEqual(p.updates,3)
            b=SimpleNamespace(output_root=Path(tmp)/'fresh',steps=8)
            fresh=runner.optimize_bound(b,w,row,seed,0,'h',4.,.1,Progress())
            self.assertEqual(fresh,resumed)
            _,final=runner.bound_paths(a,row,4.)
            final.unlink()  # Only this test's disposable result, never user data.
            p=Progress(); replay=runner.optimize_bound(a,w,row,seed,0,'h',4.,.1,p)
            self.assertEqual(replay,resumed); self.assertEqual(p.updates,0)
            with self.assertRaises(RuntimeError):runner.cached_steps(a,[row],4.,'wrong-contract')


if __name__=='__main__': unittest.main()

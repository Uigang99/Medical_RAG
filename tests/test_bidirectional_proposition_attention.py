"""CPU numerical/causality tests with real tiny Llama; no GPU experiment."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_attention_oracle import fixture, Progress
from medrag.generation.bidirectional_proposition_attention import (
    BidirectionalPropositionModel,GraphAttentionContext,allowed_chunk,graph_attention,
)
from medrag.training.attention_oracle import score,optimize_step
from medrag.training.bidirectional_attention_diagnostic import summarize
from scripts import pilot_rag2_bidirectional_attention as runner


def setup():
    legacy,row=fixture()
    row=copy.deepcopy(row);row['pairs']=[[1,0],[2,1]];row['pair_roles']=[[0,1,0,0],[0,0,1,0]]
    w=BidirectionalPropositionModel(legacy.llm,row,query_chunk=7)
    return w,row


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_mask_future_neighbors_only_and_dense_attention_reference(self):
        torch.manual_seed(1)
        tokens=torch.full((24,),3,dtype=torch.long)
        for i,(a,b) in enumerate(((2,5),(7,11),(13,16))):tokens[a:b]=i
        links=torch.zeros(4,4,dtype=torch.bool)
        for a,b in ((0,1),(1,0),(1,2),(2,1)):links[a,b]=True
        bias=torch.tensor([[0.,-.7,0.,0.],[.3,0.,.8,0.],[0.,-.2,0.,0.],[0.,0.,0.,0.]])
        ctx=GraphAttentionContext(torch.tensor([22]),(tokens,),(bias,),7,(),(links,))
        actual=allowed_chunk(ctx,0,24,'cpu')[0,0]
        expected=torch.tensor([[j<22 and (j<=i or bool(links[tokens[i],tokens[j]])) for j in range(24)] for i in range(24)])
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(actual[2,7]);self.assertFalse(actual[2,13])
        self.assertFalse(actual[2,3]);self.assertFalse(actual[2,18]);self.assertFalse(actual[:,23].any())
        q=torch.randn(1,4,24,4);k=torch.randn(1,2,24,4);v=torch.randn(1,2,24,4)
        module=SimpleNamespace(layer_idx=12)
        out,_=graph_attention(module,q,k,v,None,.5,proposition_context=ctx)
        dense_bias=bias[tokens[:,None],tokens[None,:]].masked_fill(~expected,-torch.inf)
        weights=(q@k.repeat_interleave(2,dim=1).transpose(-2,-1)*.5+dense_bias).softmax(-1)
        reference=(weights@v.repeat_interleave(2,dim=1)).transpose(1,2)
        torch.testing.assert_close(out,reference,atol=1e-6,rtol=1e-5)

    def test_real_causal_parity_bidirectional_gradients_and_frozen_target(self):
        w,row=setup();original=copy.deepcopy(row)
        weights={k:v.clone() for k,v in w.llm.state_dict().items()}
        with patch('torch.cuda.get_device_name',return_value='CPU test only'),patch('torch.cuda.max_memory_allocated',return_value=0):
            safety=runner.safety(w,row,0)
        self.assertTrue(safety['causal_legacy_parity'])
        self.assertTrue(safety['target_frozen'])
        self.assertGreater(safety['bidirectional']['reverse_gradient_norm'],0)
        self.assertGreater(safety['bidirectional']['forward_gradient_norm'],0)
        self.assertEqual(w.controller.values.shape,(4,3))
        self.assertEqual(row,original)
        for k,v in w.llm.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in w.llm.parameters()))

    def test_two_hop_future_information_flows_but_question_and_gold_do_not_leak(self):
        w,row=setup();w.configure(row,True)
        captured={12:[],13:[]}
        handles=[w.llm.model.layers[i].register_forward_hook(lambda m,a,out,i=i:captured[i].append(out.detach().clone())) for i in captured]
        try:
            with torch.no_grad():
                first=w([row],0).last_hidden
                changed=copy.deepcopy(row);changed['input_ids'][13:16]=[71,72,73]
                w([changed],0)
                question=copy.deepcopy(row);question['input_ids'][18:20]=[81,82]
                w([question],0)
                label=copy.deepcopy(row);label['gold_index']=1
                label_hidden=w([label],0).last_hidden
            # A cannot see C directly at the first opened layer, but via B at the next.
            torch.testing.assert_close(captured[12][0][:,2:5],captured[12][1][:,2:5],rtol=0,atol=0)
            self.assertGreater(float((captured[13][0][:,2:5]-captured[13][1][:,2:5]).abs().max()),1e-7)
            for i in captured:
                torch.testing.assert_close(captured[i][0][:,:16],captured[i][2][:,:16],rtol=0,atol=0)
            torch.testing.assert_close(first,label_hidden,rtol=0,atol=0)
        finally:
            for handle in handles:handle.remove()

    def test_full_four_conditions_and_graphless_resume(self):
        w,row=setup();empty=copy.deepcopy(row);empty.update(sample_id='medqa:empty',pairs=[],pair_roles=[])
        rows=[row,empty];prior={r['sample_id']:dict(base=dict(correct=False)) for r in rows}
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp),steps=4)
            p=Progress();zero=runner.measure_zero(a,w,rows,prior,0,'h',p)
            records={r['sample_id']:{} for r in rows}
            for mode in ('causal','bidirectional'):
                for r in rows:
                    rec=runner.optimize(a,w,r,zero[r['sample_id']],0,'h',mode,p)
                    records[r['sample_id']][mode]=rec
                    self.assertTrue(rec['complete'])
                    self.assertEqual(rec['updates'],4 if r['pairs'] else 0)
                    self.assertEqual(set(rec['milestones']),{'0','2','4'})
                    self.assertEqual(rec['initial'],zero[r['sample_id']][mode])
                self.assertEqual(runner.cached_updates(a,rows,mode,'h'),4)
            self.assertEqual(zero[empty['sample_id']]['causal'],zero[empty['sample_id']]['bidirectional'])
            report=summarize(rows,records,4,'h');runner.atomic_json(Path(tmp)/'SUMMARY.json',report)
            self.assertEqual(report['n'],2);self.assertEqual(report['missing'],0)
            p=Progress()
            for mode in ('causal','bidirectional'):
                for r in rows:self.assertEqual(runner.optimize(a,w,r,zero[r['sample_id']],0,'h',mode,p),records[r['sample_id']][mode])
            self.assertEqual(p.updates,0)

    def test_partial_optimizer_resume_matches_fresh(self):
        w,row=setup();w.configure(row,True)
        zero=dict(bidirectional=score(w,row,0),source_initially_correct=False)
        class Interrupted(Progress):
            def update(self,n=1):
                super().update(n)
                if self.updates==7:raise RuntimeError('simulated interruption')
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp)/'resume',steps=8)
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                runner.optimize(a,w,row,zero,0,'h','bidirectional',Interrupted())
            self.assertEqual(runner.cached_updates(a,[row],'bidirectional','h'),5)
            p=Progress();resumed=runner.optimize(a,w,row,zero,0,'h','bidirectional',p)
            self.assertEqual(p.updates,3)
            fresh=runner.optimize(SimpleNamespace(output_root=Path(tmp)/'fresh',steps=8),w,row,zero,0,'h','bidirectional',Progress())
            self.assertEqual(resumed,fresh)
            _,final=runner.paths(a,row,'bidirectional');final.unlink()
            p=Progress();self.assertEqual(runner.optimize(a,w,row,zero,0,'h','bidirectional',p),resumed)
            self.assertEqual(p.updates,0)


if __name__=='__main__':unittest.main()

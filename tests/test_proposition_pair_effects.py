"""CPU tests through real intervention, supervision, resume and QA paths."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM

from medrag.generation.proposition_kg_attention import PropositionPairController,PropositionAttentionModel
from medrag.training.proposition_pair_effects import (
    PairEffectController,_raw_pair_logits,actions,select_cohort,sampled_pairs,
    supervision_loss,training_constants,local_metrics,signal_report,paired_interval,
)

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('effect_pilot',ROOT/'scripts/pilot_rag2_proposition_pair_effects.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class Progress:
    def __init__(self):
        self.names=list(runner.STAGES); self.weights=[None]*len(self.names); self.index=0
    def stage(self,index,total,**kw):
        import time
        self.index=index; self.stage_started=time.monotonic()
    def update(self,*a): pass
    def complete(self,*a): pass
    def log(self,*a): pass


def fixture():
    torch.manual_seed(31)
    cfg=LlamaConfig(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=6,
                    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.)
    llm=LlamaForCausalLM(cfg).eval()
    c=PairEffectController(32,hidden=12,delta=.5,floor=.001)
    wrapper=PropositionAttentionModel(llm,c,cut_layer=2,attention_layers=(2,3,4,5),query_chunk=7,
                                     gradient_checkpointing=False).eval()
    row=dict(sample_id='q1',dataset='medqa',split='train',pilot_split='train',
        input_ids=list(range(1,21)),proposition_spans=[[1,4],[5,8],[9,12]],question_span=[13,16],
        pairs=[[1,0],[2,0],[2,1]],pair_roles=[[1,0,0,0],[0,1,0,0],[0,0,1,0]],
        proposition_ids=['p0','p1','p2'],choice_token_ids=[31,32,33,34],gold_index=0,option_labels=list('ABCD'))
    return wrapper,row


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_trunk_matches_original_architecture(self):
        torch.manual_seed(7)
        net=PropositionPairController(8,hidden=12,layers=8)
        torch.nn.init.normal_(net.output.weight,0,.2)
        q=torch.randn(16); p=torch.randn(3,16); pairs=torch.tensor([[1,0],[2,1]])
        roles=torch.tensor([[1.,0,0,0],[0,1.,0,0]])
        expected=net(q,p,pairs,roles)
        actual=net.max_bias*_raw_pair_logits(net,q,p,pairs,roles).tanh()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_exact_single_pair_layer_and_empty_graph(self):
        w,r=fixture(); c=w.controller
        q=torch.randn(64); p=torch.randn(3,64); pairs=torch.tensor(r['pairs']); roles=torch.tensor(r['pair_roles'])
        c.mode='intervention'; c.intervention=(1,2,-1)
        b=c(q,p,pairs,roles)
        expected=torch.zeros(3,4); expected[1,2]=-.5
        torch.testing.assert_close(b,expected)
        c.mode='learned'
        self.assertEqual(tuple(c(q,p,pairs[:0],roles[:0]).shape),(0,4))
        self.assertTrue(c.enabled)

    def test_actions_abstain_and_use_both_directions(self):
        x=torch.tensor([[.02,-.2],[-.1,.3],[.001,.002],[-.1,-.2]])
        self.assertEqual(actions(x,.01).tolist(),[0,2,1,1])

    def test_cohort_is_label_independent_and_disjoint(self):
        rows=[]
        for i in range(200):
            rows.append(dict(sample_id=str(i),split='train' if i<100 else 'validation',
                dataset='medqa' if i%3==0 else 'medmcqa',input_ids=[i],question_span=[0,1],gold_index=i%4))
        chosen=select_cohort(rows,train_n=16,dev_n=8,heldout_n=8)
        changed=[dict(r,gold_index=(r['gold_index']+1)%4) for r in rows]
        other=select_cohort(changed,train_n=16,dev_n=8,heldout_n=8)
        self.assertEqual([r['sample_id'] for r in chosen],[r['sample_id'] for r in other])
        self.assertEqual(len({r['sample_id'] for r in chosen}),32)

    def test_real_measurement_cache_supervision_and_joint_paths(self):
        w,row=fixture()
        with tempfile.TemporaryDirectory() as directory:
            a=SimpleNamespace(output_root=Path(directory),seed=5,pairs_per_question=2,delta=.5,
                 effect_floor=.001,numerical_ce_tolerance=1e-5,learning_rate=.001,
                 epochs=2,batch_questions=1,heldout_questions=1)
            record=runner.measure_one(a,w,row,0,'contract')
            self.assertEqual(len(record['measurements']),2)
            self.assertEqual(record['numerical_ce_difference'],0.)
            for m in record['measurements']:
                self.assertEqual(len(m['effects']),4)
                for layer in range(4):
                    for direction in range(2):
                        self.assertAlmostEqual(m['effects'][layer][direction],record['base']['ce']-m['raw'][layer][direction]['ce'])
            self.assertEqual(set(record['features']),{'q','p','pairs','roles'})
            with patch.object(runner,'target_score',side_effect=AssertionError('cache must not regenerate')):
                cached=runner.measure_one(a,w,row,0,'contract')
            self.assertEqual(cached['base'],record['base'])
            c=w.controller
            opt=torch.optim.AdamW(c.parameters(),lr=.001)
            # A random tiny Llama can have only sub-floor interventions. Use an
            # explicit synthetic nonzero target for the regression-gradient test.
            synthetic=copy.deepcopy(record)
            synthetic['measurements'][0]['effects'][0]=[.05,-.04]
            runner.update_controller(c,opt,[synthetic])
            self.assertGreater(c.output.weight.grad.abs().sum().item(),0)
            self.assertTrue(all(p.grad is None for p in w.llm.parameters()))
            constants,scale=training_constants([record])
            fitted=runner.fit(a,32,torch.device('cpu'),[record],[record],constants,scale,'contract',Progress())
            with patch.object(runner,'update_controller',side_effect=AssertionError('completed fit must resume')):
                resumed=runner.fit(a,32,torch.device('cpu'),[record],[record],constants,scale,'contract',Progress())
            for x,y in zip(fitted.parameters(),resumed.parameters()): torch.testing.assert_close(x,y)
            w.controller=fitted
            joint=runner.joint_eval(a,w,[row],0,'contract',Progress())
            self.assertEqual(len(joint),3)
            self.assertEqual({x['condition'] for x in joint},{'zero','constant','learned'})
            signal=signal_report([record],a.effect_floor)
            report=runner.report(a,fitted,[record],joint,signal,'contract')
            self.assertEqual(report['joint']['zero']['n'],1)
            self.assertEqual(report['local']['questions'],1)
            # All aggregate values must serialize without numpy-bool surprises.
            import json
            json.dumps(report)

    def test_changed_label_cannot_reuse_measurements(self):
        w,row=fixture()
        with tempfile.TemporaryDirectory() as directory:
            a=SimpleNamespace(output_root=Path(directory),seed=1,pairs_per_question=1,numerical_ce_tolerance=1e-5)
            runner.measure_one(a,w,row,0,'x')
            with self.assertRaises(RuntimeError): runner.measure_one(a,w,dict(row,gold_index=1),0,'x')

    def test_answer_suffix_never_enters_features(self):
        w,row=fixture(); c=w.controller; c.capture=True
        runner.target_score(w,row,0); first=c.captured
        other=copy.deepcopy(row); other['input_ids'][16:]=[80,81,82,83]; other['gold_index']=3
        runner.target_score(w,other,0)
        for k in first: torch.testing.assert_close(first[k],c.captured[k],atol=0,rtol=0)

    def test_question_level_interval(self):
        self.assertEqual(paired_interval([.2]*8),[.2,.2])


if __name__=='__main__': unittest.main()

"""Small CPU Llama tests through measurement, probe, selection and joint paths."""
from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM

from medrag.generation.proposition_kg_attention import PropositionAttentionModel,validate_layout
from medrag.training.attention_intervention_diagnostic import (
    GROUPS,ACTIONS,InterventionController,AttentionProbe,install_probe,candidates,candidate_row,
    reverse_propositions,cohort,gains,EffectPredictor,select_actions,features,
)

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagnostic_runner',ROOT/'scripts/pilot_rag2_attention_identifiability.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def fixture():
    torch.manual_seed(18)
    llm=LlamaForCausalLM(LlamaConfig(vocab_size=101,hidden_size=16,intermediate_size=32,
        num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128,attention_dropout=0.)).eval()
    controller=InterventionController(16,layers=12)
    model=PropositionAttentionModel(llm,controller,cut_layer=12,
        attention_layers=tuple(i for g in GROUPS for i in g),query_chunk=7,gradient_checkpointing=False).eval()
    probe=AttentionProbe();install_probe(probe)
    row=dict(sample_id='medqa:q1',dataset='medqa',split='train',diagnostic_split='capability',
             input_ids=list(range(1,25)),proposition_spans=[[2,5],[7,11],[13,16]],question_span=[18,20],
             pairs=[[1,0]],pair_roles=[[0,1,0,0]],proposition_ids=['A','B','C'],
             proposition_document_ids=['d1','d1','d2'],choice_token_ids=[41,42,43,44],gold_index=0)
    return model,probe,row


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_matching_and_reverse_keep_text_and_causal_pairs(self):
        _,_,row=fixture();items=candidates(row,4,7)
        self.assertEqual([x['kind'] for x in items],['graph','non_graph'])
        self.assertNotIn(items[1]['pair'],row['pairs'])
        moved,mi=reverse_propositions(row,items)
        validate_layout(candidate_row(moved,mi))
        self.assertEqual(row['input_ids'][18:],moved['input_ids'][18:])
        self.assertEqual(moved['proposition_ids'],['C','B','A'])
        self.assertEqual(moved['pair_roles'],[[0,0,1,0]])
        twice,ti=reverse_propositions(moved,mi)
        self.assertEqual(twice['input_ids'],row['input_ids'])
        self.assertEqual(ti[0]['pair'],items[0]['pair'])

    def test_fresh_nested_cohort_excludes_old_stems(self):
        rows=[]
        for ds in ('medmcqa','medqa'):
            for i in range(60):
                rows.append(dict(sample_id=f'{ds}:{i}',dataset=ds,split='train',
                    input_ids=[i+(100 if ds=='medqa' else 0)],question_span=[0,1]))
        sizes=dict(capability=8,dev=8,heldout=8,train=8)
        a=cohort(rows,{'medqa:0'},5,sizes)
        b=cohort(rows,{'medqa:0'},5,{**sizes,'train':16})
        self.assertEqual([r['sample_id'] for r in a if r['diagnostic_split']!='train'],
                         [r['sample_id'] for r in b if r['diagnostic_split']!='train'])
        self.assertNotIn('medqa:0',{r['sample_id'] for r in a})
        self.assertEqual(len({tuple(r['input_ids']) for r in a}),len(a))

    def test_real_forward_probe_dose_and_cache(self):
        w,probe,row=fixture();items=candidates(row,1,7)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output_root=Path(tmp),noise_tolerance=1e-5)
            result=runner.measure_one(args,w,probe,row,items,0,'h',capture_mass=True)
            self.assertEqual(gains(result).shape,(2,18))
            self.assertEqual(len(result['attention_mass']),6)
            for p in result['attention_mass']:
                self.assertGreater((p['changed']['mass']-p['zero']['mass'])*p['dose'],0)
            cached=runner.measure_one(args,w,probe,row,items,0,'h',capture_mass=True)
            self.assertEqual(cached['scores'],result['scores'])
            with self.assertRaises(RuntimeError):runner.read_tensor(runner.cache_path(args,row),'wrong')
            # Both actions are applied; first/second layer groups have disjoint columns.
            runner.set_actions(w,[(0,0,2.),(1,2,-1.)])
            b=w.controller(torch.zeros(32),torch.zeros(3,32),torch.tensor([[1,0],[2,0]]),torch.zeros(2,4))
            self.assertTrue(torch.equal(b[0,:4],torch.full((4,),2.)))
            self.assertTrue(torch.equal(b[1,8:],torch.full((4,),-1.)))
            self.assertEqual(float(b[0,4:].abs().sum()),0.)
            outcome=runner.score(w,candidate_row(row,items),0)
            self.assertTrue(np.isfinite(outcome['ce']))
            runner.set_actions(w,[])
            with torch.no_grad():
                expected=w.zero_bias_reference(candidate_row(row,items))
                actual=w([candidate_row(row,items)],pad_id=0).last_hidden
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_gold_and_suffix_do_not_enter_predictor_features(self):
        w,probe,row=fixture();items=candidates(row,1,7)
        w.controller.capture=True
        runner.score(w,candidate_row(row,items),0)
        before=features(w.controller.captured,row,items)
        changed={**row,'input_ids':row['input_ids'][:20]+[0]*4,'gold_index':3}
        runner.score(w,candidate_row(changed,items),0)
        after=features(w.controller.captured,changed,items)
        for k in before:torch.testing.assert_close(before[k],after[k],atol=0,rtol=0)
        model=EffectPredictor(32)
        loss=model(before).square().mean()+model(before).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in w.llm.parameters()))

    def test_noop_distinct_pair_and_kind(self):
        items=[dict(kind='graph'),dict(kind='graph'),dict(kind='non_graph')]
        values=np.zeros((3,len(ACTIONS)))
        self.assertEqual(select_actions(values,items,'graph',2,.01),[])
        values[0,1]=.1;values[0,2]=.09;values[1,0]=.05;values[2,0]=.8
        self.assertEqual(select_actions(values,items,'graph',2,.01),[(0,1),(1,0)])
        self.assertEqual(select_actions(values,items,'non_graph',1,.01),[(2,0)])

    def test_joint_reporting_and_resumption_use_real_call_path(self):
        w,probe,row=fixture()
        row.update(input_ids=list(range(1,31)),proposition_spans=[[2,5],[7,10],[12,15],[17,20]],
                   question_span=[22,25],pairs=[[1,0],[2,0]],pair_roles=[[1,0,0,0],[0,1,0,0]],
                   proposition_ids=['A','B','C','D'],proposition_document_ids=['d1','d1','d2','d2'])
        items=candidates(row,2,7)
        class Progress:
            def stage(self,*args,**kwargs):pass
            def update(self,*args):pass
            def complete(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp),noise_tolerance=1e-5,effect_floor=.01,seed=7)
            r=runner.measure_one(a,w,probe,row,items,0,'h')
            model=EffectPredictor(32)
            with torch.no_grad():model.head[-1].bias.fill_(.2)
            runner.save_tensor(a.output_root/'predictor/best.pt',dict(contract_sha256='h',state=model.state_dict()))
            constant=np.full(len(ACTIONS),.1)
            out=runner.joint(a,w,[row],[r],0,'h',model,1.,constant,Progress())
            two=next(v for v in out if v['mode']=='graph2')
            self.assertEqual(len(two['selected']),2)
            report=runner.final_report(a,[r],out,model,1.,constant,{},dict(passed=True),'h')
            runner.atomic_json(a.output_root/'summary.json',report)
            cached=runner.joint(a,w,[row],[r],0,'h',model,1.,constant,Progress())
            self.assertEqual(out,cached)

    def test_offline_fit_checkpoint_and_resume_on_synthetic_effects(self):
        # Synthetic labels test the training machinery, not scientific quality.
        items=[dict(kind='graph'),dict(kind='non_graph')]
        base=dict(ce=1.,correct=False,p_gold=.3)
        torch.manual_seed(8)
        records=[]
        for i in range(8):
            values=[[[dict(ce=.4 if g==0 and j==5 else 1.2) for j in range(6)]
                     for g in range(3)] for _ in range(2)]
            records.append(dict(sample_id=str(i),items=copy.deepcopy(items),base=base,
                                baselines=[base]*3,scores=values,
                                features=dict(x=torch.randn(2,6,32),aux=torch.randn(2,12))))
        class Progress:
            def __init__(self):self.names=['']*13;self.weights=[None]*13
            def stage(self,*args,**kwargs):pass
            def update(self,*args):pass
            def complete(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp),effect_floor=.01,seed=7,tiny_steps=60,
                              epochs=2,learning_rate=.01)
            model,scale,constant,tiny=runner.fit(a,records,records,'h','cpu',Progress())
            self.assertTrue(tiny['passed'])
            self.assertIsNotNone(model)
            before=runner.predictions(model,records[0],scale)
            resumed,*_=runner.fit(a,records,records,'h','cpu',Progress())
            np.testing.assert_array_equal(before,runner.predictions(resumed,records[0],scale))


if __name__=='__main__':unittest.main()

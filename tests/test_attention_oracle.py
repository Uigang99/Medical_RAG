"""CPU tiny-Llama tests through the real exhaustive, backward and resume paths."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.proposition_kg_attention import PropositionAttentionModel, validate_layout
from medrag.training.attention_intervention_diagnostic import GROUPS, ACTIONS
from medrag.training.attention_oracle import OracleBiasTable, install_oracle_attention, score, single_table, optimize_step
from scripts import pilot_rag2_attention_oracle as runner


class Progress:
    def __init__(self):
        self.names=['']*7; self.weights=[None]*7; self.updates=0
    def stage(self,*args,**kwargs): pass
    def update(self,n=1): self.updates+=n
    def complete(self,*args,**kwargs): pass


def fixture():
    torch.manual_seed(18)
    llm=LlamaForCausalLM(LlamaConfig(vocab_size=101,hidden_size=16,intermediate_size=32,
        num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128,attention_dropout=0.)).eval()
    row=dict(sample_id='medqa:q1',dataset='medqa',split='train',diagnostic_split='capability',
        input_ids=list(range(1,25)),proposition_spans=[[2,5],[7,11],[13,16]],question_span=[18,20],
        pairs=[[1,0],[2,0],[2,1]],pair_roles=[[0,1,0,0]]*3,
        choice_token_ids=[41,42,43,44],gold_index=0)
    controller=OracleBiasTable(16,row['pairs'])
    w=PropositionAttentionModel(llm,controller,attention_layers=tuple(i for g in GROUPS for i in g),
        gradient_checkpointing=True,query_chunk=7).eval()
    install_oracle_attention()
    return w,row


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_causal_full_pair_table_distinct_signs_and_groups(self):
        w,row=fixture()
        values=torch.tensor([[1.,-1.,.5],[-.5,2.,0.],[.1,.2,-2.]])
        w.controller.assign(values)
        result=w.controller(torch.zeros(32),torch.zeros(3,32),torch.tensor(row['pairs']),torch.zeros(3,4))
        self.assertEqual(result.shape,(3,12))
        torch.testing.assert_close(result[:,::4],values)
        with self.assertRaises(RuntimeError):
            w.controller(torch.zeros(32),torch.zeros(3,32),torch.tensor([[0,1],[2,0],[2,1]]),torch.zeros(3,4))
        invalid=copy.deepcopy(row);invalid['pairs'][0]=[0,1]
        with self.assertRaises(ValueError): validate_layout(invalid)

    def test_frozen_target_and_real_mask_gradient(self):
        w,row=fixture(); before={k:v.clone() for k,v in w.llm.state_dict().items()}
        check=runner.safety(w,row,0)
        self.assertTrue(check['target_frozen'])
        self.assertLess(check['grad_forward_max_logit_delta'],1e-4)
        self.assertGreater(check['optimizer_probe']['gradient_norm'],0)
        initial=score(w,row,0)
        opt=torch.optim.Adam([w.controller.values],lr=.1)
        for _ in range(5):
            _,_,d=optimize_step(w,row,0,opt,2.)
            self.assertGreater(d['gradient_norm'],0)
        final=score(w,row,0)
        self.assertLess(final['ce'],initial['ce'])
        self.assertLessEqual(float(w.controller.values.detach().abs().max()),2.)
        for k,v in w.llm.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_exhaustive_joint_report_and_resume_without_gate(self):
        w,row=fixture(); prior=score(w,row,0)
        old={row['sample_id']:dict(base=prior,items=[dict(kind='graph',pair=[1,0])])}
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp),joint_steps=3,learning_rate=.1,seed=31)
            progress=Progress()
            singles=runner.measure_singles(a,w,[row],old,0,'h',progress)
            self.assertEqual(len(singles[row['sample_id']]['records']),3)
            self.assertEqual(singles[row['sample_id']]['summary']['sampled_pairs'],1)
            self.assertEqual(sum(len(r['scores']) for r in singles[row['sample_id']]['records']),3*len(ACTIONS))
            combined=runner.measure_combinations(a,w,[row],singles,0,'h',progress)
            joint=runner.optimize_all(a,w,[row],singles,combined,0,'h',progress)
            self.assertEqual(len(joint[row['sample_id']]['restarts']),2)
            self.assertEqual(len(joint[row['sample_id']]['values']),3)
            self.assertLessEqual(joint[row['sample_id']]['best']['ce'],singles[row['sample_id']]['base']['ce']+1e-6)
            report=runner.summarize([row],singles,combined,joint,'h')
            runner.atomic_json(Path(tmp)/'SUMMARY.json',report)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['missing'],0)
            # Resume each exact workflow path without changing cached values.
            p=Progress()
            again=runner.measure_singles(a,w,[row],old,0,'h',p)
            c2=runner.measure_combinations(a,w,[row],again,0,'h',p)
            j2=runner.optimize_all(a,w,[row],again,c2,0,'h',p)
            self.assertEqual(joint,j2)
            self.assertEqual(p.updates,0)
            # Remove ONLY this temporary final to exercise completed optimizer checkpoint replay.
            (runner.qdir(a,row)/'joint.json').unlink()
            j3=runner.optimize_all(a,w,[row],again,c2,0,'h',p)
            self.assertEqual(joint,j3)
            with self.assertRaises(RuntimeError):runner.read_json(Path(tmp)/'SUMMARY.json','changed')

    def test_single_grid_and_joint_are_not_the_same_parameterization(self):
        values=single_table(3,1,0)
        self.assertEqual(int(torch.count_nonzero(values)),1)
        self.assertEqual(float(values[1,0]),-2.)
        # Joint has independent values for all 3 pairs x 3 groups, not one global scalar.
        w,row=fixture()
        self.assertEqual(w.controller.values.numel(),9)

    def test_partial_optimizer_window_resume_matches_uninterrupted(self):
        w,row=fixture();base=score(w,row,0)
        # Same-shaped finite records; real Llama is used for every joint update.
        records=[dict(pair_index=i,pair=p,scores=[base]*len(ACTIONS)) for i,p in enumerate(row['pairs'])]
        single=dict(base=base,records=records,summary=runner.single_summary(base,records,[[1,0]]))
        class Interrupted(Progress):
            def update(self,n=1):
                super().update(n)
                if self.updates==7: raise RuntimeError('simulated interruption after durable step5')
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_root=Path(tmp)/'resume',joint_steps=8,learning_rate=.1,seed=31)
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                runner.optimize_question(a,w,row,single,[],0,'h',Interrupted())
            resumed=runner.optimize_question(a,w,row,single,[],0,'h',Progress())
            a.output_root=Path(tmp)/'fresh'
            fresh=runner.optimize_question(a,w,row,single,[],0,'h',Progress())
            self.assertEqual(resumed,fresh)


if __name__=='__main__': unittest.main()

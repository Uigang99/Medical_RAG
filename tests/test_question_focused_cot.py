import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from test_question_only_pced import Tokenizer, row
from medrag.generation.question_only_pced import BatchedPCED, RATIONALE_SYSTEM, render_prompt
from medrag.generation.question_only_cot import BatchedCoT, render_prompt as old_baseline
from medrag.generation.question_focused_cot import (
    NO_RAG_SYSTEM, RAG_SYSTEM, QuestionFocusedCoT, QuestionFocusedPCED,
    baseline_prompt, pced_prompt, replace_system)
from evaluate_question_focused_cot import ARMS, arguments, load_done, prompt_changes
from medrag.training.graph_input_data import atomic_jsonl


class TestFocusedCoT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=2048,eos_token_id=1,pad_token_id=0)).eval()
        cls.tok = Tokenizer()
        cls.labels = [65,66,67,68]

    def test_only_system_changes_and_passage_is_not_rewritten(self):
        r=row('a');r['documents'][0]['text']=RATIONALE_SYSTEM
        for arm,system in zip(ARMS[:2],[NO_RAG_SYSTEM,RAG_SYSTEM]):
            before=old_baseline(self.tok,r,arm);after=baseline_prompt(self.tok,r,arm)
            self.assertEqual(before.partition(RATIONALE_SYSTEM)[2],after.partition(system)[2])
        self.assertNotIn('passages',NO_RAG_SYSTEM)
        self.assertNotIn('Evidence:',baseline_prompt(self.tok,r,ARMS[0]))
        self.assertIn(RATIONALE_SYSTEM,baseline_prompt(self.tok,r,ARMS[1]))
        self.assertEqual(baseline_prompt(self.tok,r,ARMS[0]),pced_prompt(self.tok,r,None))
        one=copy.deepcopy(r);one['documents']=one['documents'][:1]
        self.assertEqual(baseline_prompt(self.tok,one,ARMS[1]),pced_prompt(self.tok,r,r['documents'][0]))
        with self.assertRaises(ValueError):replace_system('unexpected','x')

    def test_question_major_amateur_expert_prompt_assignment(self):
        rows=[row('a'),row('b')]
        prompts=[render_prompt(self.tok,r,d,'rationale') for r in rows for d in [None,*r['documents']]]
        engine=QuestionFocusedPCED(self.model,self.tok,self.labels,max_rationale_tokens=3)
        with patch.object(BatchedPCED,'prefill',return_value='captured') as capture:
            self.assertEqual(engine.prefill(prompts,use_cache=True),'captured')
        actual=capture.call_args.args[0]
        expected=[pced_prompt(self.tok,r,d) for r in rows for d in [None,*r['documents']]]
        self.assertEqual(actual,expected)
        with self.assertRaises(ValueError):engine.generate(rows,'direct')
        with self.assertRaises(ValueError):engine.prefill(prompts[:8])
        bad=row('bad');bad['documents'].pop()
        with self.assertRaises(ValueError):engine.generate([bad])

    def test_no_decoder_change_when_prompt_change_disabled(self):
        rows=[row('a'),row('b')];rows[1]['question']='Longer input?'
        kwargs={'max_rationale_tokens':3}
        for arm in ARMS[:2]:
            old=BatchedCoT(self.model,self.tok,self.labels,**kwargs).generate_plain(rows,arm)
            engine=QuestionFocusedCoT(self.model,self.tok,self.labels,arm=arm,**kwargs)
            with patch('medrag.generation.question_focused_cot.replace_system',side_effect=lambda p,s:p):
                self.assertEqual(engine.generate_plain(rows,arm),old)
            with self.assertRaises(ValueError):engine.generate_plain(rows,'wrong')
        old=BatchedPCED(self.model,self.tok,self.labels,**kwargs).generate(rows,'rationale')
        with patch('medrag.generation.question_focused_cot.replace_system',side_effect=lambda p,s:p):
            actual=QuestionFocusedPCED(self.model,self.tok,self.labels,**kwargs).generate(rows)
        self.assertEqual(actual,old)

    def test_actual_short_prompt_all_three_cpu_generation_paths(self):
        rows=[row('a'),row('b')];rows[1]['question']='Longer input?'
        for arm in ARMS:
            if arm=='pced_cot':
                engine=QuestionFocusedPCED(self.model,self.tok,self.labels,max_rationale_tokens=3)
                generate=lambda rs:engine.generate(rs)
            else:
                engine=QuestionFocusedCoT(self.model,self.tok,self.labels,arm=arm,max_rationale_tokens=3)
                generate=lambda rs:engine.generate_plain(rs,arm)
            batch={v['sample_id']:v for v in generate(rows)}
            for r in rows:
                serial=generate([r])[0];v=batch[r['sample_id']]
                for key in ('answer','rationale_token_ids','finish_reason','document_ids'):
                    self.assertEqual(v[key],serial[key])
                self.assertIn(v['answer'],'ABCD')
                self.assertLessEqual(len(v['rationale_token_ids']),3)
                self.assertEqual(len(v['document_ids']),0 if arm==ARMS[0] else 8)

    def test_prompt_delta_macro3_and_transitions(self):
        rows=[row(str(i)) for i in range(6)]
        for r,d in zip(rows,['medmcqa','medqa','mmlu_anatomy']+['mmlu_clinical_knowledge']*3):r['dataset']=d
        gold={r['sample_id']:'A' for r in rows}
        def result(indices):
            return {a:{str(i):{'answer':'A' if i in indices else 'B'} for i in range(6)} for a in ARMS}
        old,new=result({0,2}),result({0,1})
        out=prompt_changes(rows,gold,old,new,resamples=20)
        for v in out.values():
            self.assertAlmostEqual(v['old_macro3'],1.25/3)
            self.assertAlmostEqual(v['new_macro3'],2/3)
            self.assertAlmostEqual(v['macro3_delta'],.25)
            self.assertEqual(v['transitions']['C->W'],1)
            self.assertEqual(v['transitions']['W->C'],1)
            self.assertEqual(v['micro_delta'],0)
        self.assertEqual(out,prompt_changes(rows,gold,old,new,resamples=20))

    def test_pced_resume_contract_validation(self):
        r=row('a');saved={'sample_id':'a','dataset':'mock','arm':'pced_cot','mode':'rationale',
            'answer':'A','contract_sha256':'h','document_ids':[d['corpus_id'] for d in r['documents']],
            'priors':r['priors'],'finish_reason':'eos','rationale_token_ids':[],'expert_ranks':[],'beta':.2}
        with tempfile.TemporaryDirectory() as temp:
            a=SimpleNamespace(output_root=Path(temp));directory=Path(temp)/'pced_cot/shards'
            atomic_jsonl(directory/'a.jsonl',[saved])
            self.assertEqual(load_done(a,[r],'pced_cot','h'),{'a':saved})
            with self.assertRaises(RuntimeError):load_done(a,[r],'pced_cot','other')
            atomic_jsonl(directory/'b.jsonl',[saved])
            with self.assertRaises(RuntimeError):load_done(a,[r],'pced_cot','h')
        with self.assertRaises(SystemExit):
            arguments(['--output-root',str(ROOT/'results/question_only_cot_baselines_pced_matched_v1')])


if __name__=='__main__':unittest.main()

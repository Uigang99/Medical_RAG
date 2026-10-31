import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from test_question_only_pced import Tokenizer, Cache, row
from medrag.generation.question_only_pced import BatchedPCED, render_prompt as pced_prompt
from medrag.generation.question_only_cot import BatchedCoT, render_prompt
from medrag.training.graph_input_data import atomic_jsonl
from evaluate_question_only_cot_baselines import summarize, load_done, ARMS


class ScriptedCoT(BatchedCoT):
    def __init__(self, model, tok):
        super().__init__(model, tok, [65,66,67,68], max_rationale_tokens=32)
        self.histories = {}
        self.scripts = [[1], [120]*40, tok.encode('Reason.\nFinal answer:')]

    def make_logits(self, cache):
        logits = torch.zeros(len(cache.identities),128)
        for i,q in enumerate(cache.identities):
            step = len(self.histories[q]); script = self.scripts[q]
            logits[i,65+q] = 5
            logits[i,script[min(step,len(script)-1)]] = 20
        return logits

    def prefill(self, prompts, **kwargs):
        cache = Cache(range(len(prompts)))
        self.histories = {i:[] for i in cache.identities}
        return self.make_logits(cache),cache,torch.ones(len(prompts),1,dtype=torch.long)

    def advance(self,tokens,streams,cache,mask):
        for i,q in enumerate(cache.identities):self.histories[q].append(tokens[i])
        return self.make_logits(cache),cache,torch.cat([mask,mask.new_ones(len(mask),1)],-1)


class SameStreamPCED(BatchedPCED):
    """Every expert has identical logits: reference decoding reduces to argmax."""
    def __init__(self, model,tok):
        super().__init__(model,tok,[65,66,67,68],max_rationale_tokens=32)
        self.fake = ScriptedCoT(model,tok)

    def prefill(self,prompts,**kwargs):
        logits,cache,mask = self.fake.prefill(prompts[::9])
        return logits.repeat_interleave(9,0),Cache([q for q in cache.identities for _ in range(9)]),mask.repeat_interleave(9,0)

    def advance(self,tokens,streams,cache,mask):
        unique = Cache(cache.identities[::9])
        logits,_,newmask = self.fake.advance(tokens,1,unique,mask[::9])
        return logits.repeat_interleave(9,0),cache,newmask.repeat_interleave(9,0)


class TestMatchedCoT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.model=LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=2048,eos_token_id=1,pad_token_id=0)).eval()
        cls.tok=Tokenizer()

    def test_prompts_preserve_reference_and_evidence_order(self):
        r=row('a');r['documents'][0]['title']='Title'
        self.assertEqual(render_prompt(self.tok,r,'no_rag_cot'),pced_prompt(self.tok,r,None,'rationale'))
        one=copy.deepcopy(r);one['documents']=one['documents'][:1]
        self.assertEqual(render_prompt(self.tok,one,'base_rag_cot'),pced_prompt(self.tok,r,r['documents'][0],'rationale'))
        text=render_prompt(self.tok,r,'base_rag_cot')
        self.assertEqual(text.count('evidence'),9)  # system + eight unchanged documents
        self.assertLess(text.index('[1]'),text.index('[8]'))
        with self.assertRaises(ValueError):render_prompt(self.tok,r,'wrong')

    def test_all_termination_paths_match_reference_state_machine(self):
        rows=[row(str(i)) for i in range(3)]
        engine=ScriptedCoT(self.model,self.tok)
        result={r['sample_id']:r for r in engine.generate_plain(rows,'base_rag_cot')}
        reference=SameStreamPCED(self.model,self.tok)
        expected={r['sample_id']:r for r in reference.generate(rows,'rationale')}
        for i,finish in enumerate(['eos','token_budget','natural_marker']):
            got=result[str(i)]
            self.assertEqual(got['finish_reason'],finish)
            for key in ['answer','answer_token_id','rationale_token_ids','finish_reason']:
                self.assertEqual(got[key],expected[str(i)][key])
            self.assertEqual(engine.histories[i],reference.fake.histories[i])
        self.assertNotIn(1,engine.histories[0])
        self.assertEqual(len(result['1']['rationale_token_ids']),32)
        self.assertEqual(result['2']['forced_marker_tokens'],0)

    def test_tiny_model_batched_and_serial(self):
        rows=[row('a'),row('b')];rows[1]['question']='A longer question?'
        for arm in ARMS:
            engine=BatchedCoT(self.model,self.tok,[65,66,67,68],max_rationale_tokens=3)
            batched={r['sample_id']:r for r in engine.generate_plain(rows,arm)}
            for r in rows:
                serial=engine.generate_plain([r],arm)[0]
                for key in ['answer','rationale_token_ids','finish_reason','document_ids']:
                    self.assertEqual(batched[r['sample_id']][key],serial[key])
                torch.testing.assert_close(torch.tensor(batched[r['sample_id']]['choice_scores']),
                                           torch.tensor(serial['choice_scores']),atol=1e-6,rtol=1e-5)

    def test_macro3_pooled_mmlu_and_paired_counts(self):
        rows=[row(str(i)) for i in range(6)]
        for r,d in zip(rows,['medmcqa','medqa','mmlu_anatomy']+['mmlu_clinical_knowledge']*3):r['dataset']=d
        gold={r['sample_id']:'A' for r in rows}
        results={}
        for name,correct in zip([*ARMS,'pced_cot'],[{0,2},{0,1},{0,1,2}]):
            results[name]={r['sample_id']:{'answer':'A' if i in correct else 'B',
                'finish_reason':'natural_marker','rationale_token_ids':[120]} for i,r in enumerate(rows)}
        summary=summarize(rows,gold,results,resamples=20)
        self.assertAlmostEqual(summary['conditions'][ARMS[0]]['macro3_accuracy'],(1+0+.25)/3)
        self.assertAlmostEqual(summary['conditions'][ARMS[0]]['micro_accuracy'],2/6)
        paired=summary['paired']['base_rag_cot_minus_no_rag_cot']
        self.assertEqual(paired['transitions']['C->W'],1)
        self.assertEqual(paired['transitions']['W->C'],1)
        self.assertAlmostEqual(paired['macro3_delta'],paired['macro3_recovery_rate']-paired['macro3_harm_rate'])
        self.assertEqual(summary,summarize(rows,gold,results,resamples=20))

    def test_resume_validates_contract_and_duplicates(self):
        rows=[row('a')];arm='no_rag_cot'
        saved={'sample_id':'a','dataset':'mock','arm':arm,'answer':'A','contract_sha256':'h',
               'document_ids':[],'finish_reason':'eos','rationale_token_ids':[]}
        with tempfile.TemporaryDirectory() as temp:
            a=SimpleNamespace(output_root=Path(temp));directory=Path(temp)/arm/'shards'
            atomic_jsonl(directory/'a.jsonl',[saved])
            self.assertEqual(load_done(a,rows,arm,'h'),{'a':saved})
            with self.assertRaises(RuntimeError):load_done(a,rows,arm,'wrong')
            atomic_jsonl(directory/'b.jsonl',[saved])
            with self.assertRaises(RuntimeError):load_done(a,rows,arm,'h')


if __name__=='__main__':unittest.main()

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from medrag.text_graph_evaluation import (CONDITIONS, generate_tokens, generation_inputs,
    pack_evaluation_pair, parse_answer, summarize)
from medrag.training.shared_medical_lora import adapt_llama


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1] + [2]*len(" ".join(m["content"] for m in messages).split())


def edge(i):
    return {"edge_id":str(i),"proposition_id":str(i),"document_id":"doc",
        "subject_entity_id":"A","object_entity_id":str(i),"subject_surface":"A",
        "object_surface":"thing "+str(i),"relation_surface":"affects","claim":"A affects thing "+str(i)+"."}


class SixConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        path=Path(__file__).parents[1]/"scripts/evaluate_rag2_text_graph_six_conditions.py"
        spec=importlib.util.spec_from_file_location("six_conditions_entry",path)
        cls.entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.entry)

    def model(self):
        torch.manual_seed(42)
        return LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=48,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,bos_token_id=1,
            eos_token_id=31,pad_token_id=0,attention_dropout=0.)).eval()

    def test_gold_independent_joint_packing(self):
        row={"sample_id":"q","question":"Which observation?","options":{"A":"one","B":"two"}}
        a=pack_evaluation_pair(Tokenizer(),row,[edge(i) for i in range(30)],90,16)
        b=pack_evaluation_pair(Tokenizer(),{**row,"answer":"very long gold label should never be read"},[edge(i) for i in range(30)],90,16)
        for k in ("graph","text_prompt_ids","graph_prompt_ids","none_prompt_ids","packing"):
            self.assertEqual(a[k],b[k])
        self.assertEqual(a["graph_prompt_ids"],a["none_prompt_ids"])
        self.assertLessEqual(max(len(a["text_prompt_ids"]),len(a["graph_prompt_ids"])+1)+16,90)
        self.assertGreater(a["packing"]["removed_propositions"],0)
        self.assertEqual({e["proposition_id"] for e in a["edges"]},{p["proposition_id"] for p in a["propositions"]})
        empty=pack_evaluation_pair(Tokenizer(),row,[],90,16)
        self.assertEqual(empty["graph"]["node_ids"],[])
        self.assertEqual(empty["text_prompt_ids"],empty["none_prompt_ids"])

    def test_greedy_embeddings_equal_native_text_and_batch_slicing(self):
        model=self.model()
        prompts=[[1,5,6,7],[1,8]]
        actual=generate_tokens(model,prompts,pad_id=0,eos_ids=[31],max_new_tokens=4)
        for ids,result in zip(prompts,actual):
            cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=4,pad_token_id=0,eos_token_id=31,use_cache=True)
            with torch.inference_mode():
                expected=model.generate(torch.tensor([ids]),generation_config=cfg)[0,len(ids):].tolist()
            self.assertEqual(result["generated_ids"],expected)

    def test_graph_inserted_after_bos_and_used_for_generation(self):
        model=self.model()
        prompts=[[1,4,5],[1,6]]
        soft=[torch.randn(1,32),torch.randn(1,32)]
        with torch.inference_mode():
            packed=generation_inputs(model,prompts,0,soft)
            torch.testing.assert_close(packed["inputs_embeds"][0,1],soft[0][0])
            torch.testing.assert_close(packed["inputs_embeds"][1,2],soft[1][0])
            torch.testing.assert_close(packed["inputs_embeds"][0,0],model.get_input_embeddings()(torch.tensor(1)))
            self.assertEqual(packed["attention_mask"].tolist(),[[1,1,1,1],[0,1,1,1]])
            mask=packed["attention_mask"]
            expected=model(inputs_embeds=packed["inputs_embeds"],attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp_min(0)).logits[:,-1].argmax(-1).tolist()
            result=generate_tokens(model,prompts,pad_id=0,eos_ids=[31],max_new_tokens=1,soft_prompts=soft)
            self.assertEqual([r["generated_ids"][0] for r in result],expected)

    def test_base_disable_lora_restores_untrained_model(self):
        base=self.model()
        x=torch.tensor([[1,3,4]])
        with torch.no_grad(): original=base(x).logits.clone()
        model=adapt_llama(base,rank=2,alpha=4).eval()
        model.gradient_checkpointing_disable()
        with torch.no_grad():
            for n,p in model.named_parameters():
                if "lora_B" in n: p.normal_(0,.2)
            trained=model.get_base_model()(x).logits
            with model.disable_adapter(): restored=model.get_base_model()(x).logits
            again=model.get_base_model()(x).logits
        self.assertFalse(torch.allclose(original,trained))
        torch.testing.assert_close(original,restored)
        torch.testing.assert_close(trained,again)

    def test_parser_and_complete_macro3_paired_counts(self):
        opts={"A":"one","B":"two"}
        self.assertEqual(parse_answer("Answer: B",opts),"B")
        self.assertEqual(parse_answer("A.",opts),"A")
        for text in ("Answer: A or B","A because ...","Answer: C",""):
            self.assertIsNone(parse_answer(text,opts))
        rows=[{"sample_id":str(i),"dataset":ds,"answer":"A"} for i,ds in enumerate(
            ["medmcqa"]*3+["medqa"]+["mmlu_anatomy"]*2)]
        predictions={}
        for name,_,_ in CONDITIONS:
            outputs=["A","B","B","A","A","B"]
            if name=="5_trained_graph": outputs=["A","A","B","A","A",None]
            predictions[name]=[{"sample_id":str(i),"prediction":v,"finish_reason":"eos"} for i,v in enumerate(outputs)]
        report=summarize(rows,predictions,bootstrap_replicates=50)
        s=report["conditions"]["3_trained_text"]
        self.assertEqual(s["micro_pct"],50.)
        self.assertAlmostEqual(s["macro3_pct"],(100/3+100+50)/3)
        self.assertEqual(report["conditions"]["5_trained_graph"]["parse_failures"],1)
        paired=report["primary_comparison"]
        self.assertEqual(paired["transitions"],{"C_C":3,"C_W":0,"W_C":1,"W_W":2})
        self.assertAlmostEqual(paired["delta_micro_pp"],100/6)
        predictions["1_base_no_rag"].pop()
        with self.assertRaises(ValueError): summarize(rows,predictions,10)

    def test_unfinished_training_fails_cheaply(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(training_root=Path(tmp),batch_size=16,embedding_batch_size=128,max_new_tokens=16)
            with self.assertRaisesRegex(RuntimeError,"training is not complete"):
                self.entry.preflight(args)

    def test_resume_checks_condition_and_exact_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"batch.json"
            p.write_text(json.dumps({"contract_sha256":"hash","condition":"base","rows":[{"sample_id":"x"}]}))
            self.assertEqual(len(self.entry.load_prediction_shard(p,["x"],"hash","base")),1)
            for ids,h,c in ((["y"],"hash","base"),(["x"],"changed","base"),(["x"],"hash","trained")):
                with self.assertRaises(RuntimeError): self.entry.load_prediction_shard(p,ids,h,c)

    def test_report_serializes_metrics_and_paired_outputs(self):
        rows=[{"sample_id":str(i),"dataset":ds,"answer":"A","graph":{"node_ids":[]},
               "packing":{"removed_propositions":0}} for i,ds in enumerate(("medmcqa","medqa","mmlu_anatomy"))]
        predictions={name:[{"sample_id":r["sample_id"],"prediction":"A","finish_reason":"eos"} for r in rows]
                     for name,_,_ in CONDITIONS}
        progress=SimpleNamespace(stage=lambda *a,**k:None,update=lambda:None,complete=lambda *a:None,log=lambda *a:None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"PAIRS_COMPLETE.json").write_text(json.dumps({"n":3}))
            self.entry.write_summary(SimpleNamespace(output_root=root),rows,predictions,"hash",progress)
            report=json.loads((root/"summary.json").read_text())
            self.assertEqual(report["subgroups"]["empty_graph"]["conditions"]["1_base_no_rag"]["accuracy_pct"],100.)
            self.assertEqual(len((root/"paired_predictions.jsonl").read_text().splitlines()),3)
            self.assertEqual(len((root/"summary.csv").read_text().splitlines()),7)
            self.assertEqual(json.loads((root/"COMPLETE.json").read_text())["conditions"],6)


if __name__=="__main__": unittest.main()

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import PeftModel

from medrag.training.shared_medical_lora import adapt_llama, adapter_state, restore_adapter, qa_losses
from medrag.training.graph_input_modules import training_attention_backend


class SharedMedicalLoRATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def base(self):
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.))
        model.config.use_cache=False
        return model

    @training_attention_backend()
    def test_lora_changes_llama_not_prompt_and_roundtrips(self):
        torch.manual_seed(42)
        original=self.base()
        frozen=copy.deepcopy(original)
        model=adapt_llama(original,rank=4,alpha=8)
        rows=[{"text_prompt_ids":[1,2,3],"graph_prompt_ids":[1,2],"target_ids":[6]},
              {"text_prompt_ids":[1,4],"graph_prompt_ids":[1,4],"target_ids":[7]}]
        saved_base={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        initial=adapter_state(model)
        model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.02)
        first=float(qa_losses(model,rows,0).mean().detach())
        for _ in range(32):
            opt.zero_grad()
            loss=qa_losses(model,rows,0).mean()
            loss.backward(); opt.step()
        model.eval()
        last=float(qa_losses(model,rows,0).mean().detach())
        self.assertLess(last,first-.05)
        self.assertTrue(all(torch.equal(p,saved_base[n]) for n,p in model.named_parameters() if not p.requires_grad))
        self.assertTrue(all('lora_' in n for n,p in model.named_parameters() if p.requires_grad))
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in adapter_state(model).items()))
        z=[torch.randn(1,32) for _ in rows]
        graph_loss=qa_losses(model,rows,0,graph_prompts=z)
        self.assertTrue(torch.isfinite(graph_loss).all())
        expected=qa_losses(model,rows,0).detach()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp,save_embedding_layers=False)
            restored=PeftModel.from_pretrained(frozen,tmp,is_trainable=False).eval()
            torch.testing.assert_close(qa_losses(restored,rows,0),expected,atol=1e-6,rtol=1e-6)
        final=adapter_state(model)
        restore_adapter(model,initial); restore_adapter(model,final)
        torch.testing.assert_close(qa_losses(model,rows,0),expected,atol=1e-6,rtol=1e-6)

    def test_wrong_checkpoint_rejected(self):
        model=adapt_llama(self.base(),rank=2,alpha=4)
        with self.assertRaises(RuntimeError): restore_adapter(model,{})

    def test_entrypoint_does_not_run_final_evaluation(self):
        source=(Path(__file__).parents[1]/"scripts/train_rag2_shared_medical_lora.py").read_text()
        self.assertNotIn('.generate(',source)
        self.assertIn('"pilot_evaluation_questions_used":0',source)


if __name__=='__main__': unittest.main()

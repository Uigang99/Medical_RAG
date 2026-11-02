from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from test_question_only_pced import Tokenizer, row
from medrag.generation import question_focused_cot as focused
from medrag.generation import restored_cot_instruction as restored
from medrag.generation.question_only_pced import BatchedPCED, RATIONALE_SYSTEM, render_prompt
from pilot_restored_cot_instruction import arguments


class TestRestoredInstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(42)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, eos_token_id=1, pad_token_id=0)).eval()
        cls.tok = Tokenizer()

    def test_exact_output_block_replacement_and_preserved_body(self):
        self.assertTrue(RATIONALE_SYSTEM.endswith(restored.OUTPUT_INSTRUCTION))
        for old, new in [(focused.NO_RAG_SYSTEM, restored.NO_RAG_SYSTEM), (focused.RAG_SYSTEM, restored.RAG_SYSTEM)]:
            self.assertEqual(old.split('Give a brief explanation')[0], new[:-len(restored.OUTPUT_INSTRUCTION)])
            self.assertNotIn('rather than a summary', new)
            self.assertTrue(new.endswith(restored.OUTPUT_INSTRUCTION))
        r = row('a'); r['documents'][0]['text'] = RATIONALE_SYSTEM
        for arm, old_system, new_system in [('no_rag_cot', focused.NO_RAG_SYSTEM, restored.NO_RAG_SYSTEM), ('base_rag_cot', focused.RAG_SYSTEM, restored.RAG_SYSTEM)]:
            old = focused.baseline_prompt(self.tok, r, arm)
            new = restored.baseline_prompt(self.tok, r, arm)
            self.assertEqual(old.replace(old_system, new_system, 1), new)
        self.assertIn(RATIONALE_SYSTEM, restored.baseline_prompt(self.tok, r, 'base_rag_cot'))

    def test_pced_stream_assignment_and_unchanged_decoder(self):
        rows = [row('a'), row('b')]
        prompts = [render_prompt(self.tok, r, d, 'rationale') for r in rows for d in [None, *r['documents']]]
        engine = restored.RestoredPCED(self.model, self.tok, [65,66,67,68], max_rationale_tokens=3)
        with patch.object(BatchedPCED, 'prefill', return_value='captured') as capture:
            self.assertEqual(engine.prefill(prompts, use_cache=True), 'captured')
        self.assertIs(capture.call_args.args[0], engine)
        self.assertEqual(capture.call_args.args[1], [restored.pced_prompt(self.tok, r, d) for r in rows for d in [None, *r['documents']]])
        with self.assertRaises(ValueError): engine.generate(rows, 'direct')
        with self.assertRaises(ValueError): engine.prefill(prompts[:8])
        for arm in ('no_rag_cot', 'base_rag_cot', 'pced_cot'):
            if arm == 'pced_cot': generate = engine.generate
            else:
                plain = restored.RestoredCoT(self.model, self.tok, [65,66,67,68], arm=arm, max_rationale_tokens=3)
                generate = lambda rs: plain.generate_plain(rs, arm)
            batch = {v['sample_id']: v for v in generate(rows)}
            for r in rows:
                serial = generate([r])[0]
                for key in ('answer', 'rationale_token_ids', 'finish_reason', 'document_ids'):
                    self.assertEqual(batch[r['sample_id']][key], serial[key])

    def test_first_final_is_not_the_same_as_empty_explanation(self):
        def v(text, ids): return {'rationale': text, 'rationale_token_ids': ids}
        final = v('Final answer:', [19918,4320,25]); reason = v('Because.\nFinal answer:', [1,19918,4320,25])
        before = {'a': final, 'b': final, 'c': reason, 'd': final}
        after = {'a': reason, 'b': final, 'c': reason, 'd': v('', [])}
        report = restored.behavior_change(before, after, 19918)
        self.assertEqual(report['before']['first_final_count'], 3)
        self.assertEqual(report['after']['first_final_count'], 1)
        self.assertEqual(report['after']['no_explanation_count'], 2)
        self.assertEqual(report['first_final_to_other'], 2)
        self.assertEqual(report['first_final_rate_delta'], -.5)
        self.assertEqual(report, restored.behavior_change(before, after, 19918))
        with self.assertRaises(ValueError): restored.behavior_change(before, {}, 19918)

    def test_defaults_and_immutable_output_guard(self):
        a = arguments([])
        self.assertEqual(a.limit, 256)
        self.assertEqual(a.output_root.name, 'restored_cot_instruction_v1_test256')
        self.assertEqual(arguments(['--limit', '64']).output_root.name, 'restored_cot_instruction_v1_test64')
        with self.assertRaises(SystemExit):
            arguments(['--output-root', str(ROOT/'results/question_focused_cot_test6545_v1')])


if __name__ == '__main__': unittest.main()

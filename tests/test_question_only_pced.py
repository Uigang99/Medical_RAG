import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from medrag.generation.question_only_pced import (
    BatchedPCED, compact_cache, initial_beta, medcpt_priors, scores, select_token, render_prompt)
from medrag.training.graph_input_data import ensure_contract
from evaluate_question_only_pced import summary, load_done, output_path


class Tokenizer:
    eos_token_id = 1
    def encode(self, s, **kwargs):
        return [ord(c) for c in s]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i > 1)
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)
    def __call__(self, texts, **kwargs):
        ids = [self.encode(s) for s in texts]
        n = max(map(len, ids))
        return {'input_ids': torch.tensor([[0]*(n-len(x))+x for x in ids]),
                'attention_mask': torch.tensor([[0]*(n-len(x))+[1]*len(x) for x in ids])}


class Cache:
    def __init__(self, identities):
        self.identities = list(identities)
    def batch_select_indices(self, indices):
        self.identities = [self.identities[i] for i in indices.tolist()]


class ScriptedEngine(BatchedPCED):
    """Deterministic EOS/budget paths; tracks the tokens committed to all streams."""
    def __init__(self, model, tok):
        super().__init__(model, tok, [65, 66, 67, 68], max_rationale_tokens=3)
        self.histories = {}
    def make_logits(self, cache):
        logits = torch.zeros(len(cache.identities), 128)
        for i, identity in enumerate(cache.identities):
            q = identity // 9
            # First question terminates immediately, second hits its token budget.
            logits[i, 1 if q == 0 else 120] = 20
            logits[i, 65+q] = 5
        return logits
    def prefill(self, prompts, **kwargs):
        cache = Cache(range(len(prompts)))
        self.histories = {i: [] for i in cache.identities}
        return self.make_logits(cache), cache, torch.ones(len(prompts), 1, dtype=torch.long)
    def advance(self, tokens, streams, cache, mask):
        for i, identity in enumerate(cache.identities):
            self.histories[identity].append(tokens[i//streams])
        mask = torch.cat([mask, mask.new_ones((len(mask), 1))], -1)
        return self.make_logits(cache), cache, mask


def row(sid):
    return {'sample_id': sid, 'dataset': 'mock', 'question': 'x?', 'options': dict(zip('ABCD', 'abcd')),
            'documents': [{'corpus_id': str(i), 'text': 'evidence'} for i in range(8)], 'priors': [.8]*8}


class TestPCED(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, eos_token_id=1, pad_token_id=0)).eval()
        cls.tok = Tokenizer()

    def test_prior_uses_all_32_and_raw_ce(self):
        docs = [{'inner_product_score': i, 'rerank_score': 0} for i in range(32)]
        priors = medcpt_priors(docs)
        self.assertEqual(len(priors), 8)
        self.assertAlmostEqual(priors[7], 2*(7/31)*.5/(7/31+.5+1e-8))
        self.assertTrue(all(0 < v < 1 for v in priors))
        for d in docs:
            d['inner_product_score'] = 2
        self.assertEqual(len(set(medcpt_priors(docs))), 1)

    def test_formula_and_max_not_average(self):
        x = torch.tensor([[[1., 2.], [4., 0.], [0., 5.]]])
        b, p = torch.tensor([.3]), torch.tensor([[.9, .6]])
        expected = torch.stack([1.3*x[0, k]-.3*x[0, 0]+2.5*p[0, k-1].log() for k in (1, 2)])
        torch.testing.assert_close(scores(x, b, p, 2.5)[0], expected)
        token, expert, values = select_token(x, b, p, 2.5)
        self.assertEqual(token.item(), expected.max(0).values.argmax().item())
        self.assertEqual(expert.item(), expected[:, token.item()].argmax().item())

    def test_beta_full_vocab_fixed_scale(self):
        logits = torch.randn(2, 1, 128).expand(-1, 9, -1)
        torch.testing.assert_close(initial_beta(logits), torch.zeros(2), atol=1e-6, rtol=0)
        b = initial_beta(torch.randn(2, 9, 128)*10)
        self.assertTrue(((b >= 0) & (b <= .693148)).all())

    def test_cached_left_padding_matches_full_forward(self):
        engine = BatchedPCED(self.model, self.tok, [65, 66, 67, 68])
        logits, cache, mask = engine.prefill(['ab', 'abcde'], use_cache=True)
        serial, _, _ = engine.prefill(['ab'], use_cache=False)
        torch.testing.assert_close(logits[0], serial[0], atol=1e-6, rtol=1e-5)
        next_logits, cache, mask = engine.advance([102, 103], 1, cache, mask)
        full, _, _ = engine.prefill(['abf', 'abcdeg'], use_cache=False)
        torch.testing.assert_close(next_logits, full, atol=1e-6, rtol=1e-5)
        compact_cache(cache, torch.tensor([1]))
        last, _, _ = engine.advance([104], 1, cache, mask[[1]])
        full, _, _ = engine.prefill(['abcdegh'], use_cache=False)
        torch.testing.assert_close(last, full, atol=1e-6, rtol=1e-5)

    def test_shared_history_eos_budget_and_compaction(self):
        engine = ScriptedEngine(self.model, self.tok)
        results = {r['sample_id']: r for r in engine.generate([row('early'), row('late')], 'rationale')}
        self.assertEqual(results['early']['finish_reason'], 'eos')
        self.assertEqual(results['late']['finish_reason'], 'token_budget')
        self.assertEqual(results['early']['answer'], 'A')
        self.assertEqual(results['late']['answer'], 'B')
        self.assertEqual(len(results['late']['rationale_token_ids']), 3)
        for base in (0, 9):
            for k in range(1, 9):
                self.assertEqual(engine.histories[base], engine.histories[base+k])
        self.assertNotIn(1, engine.histories[0])

    def test_direct_no_recurrent_generation(self):
        engine = ScriptedEngine(self.model, self.tok)
        results = engine.generate([row('a')], 'direct')
        self.assertEqual(results[0]['answer'], 'A')
        self.assertEqual(results[0]['rationale_token_ids'], [])
        self.assertTrue(all(not h for h in engine.histories.values()))

    def test_paired_summary_and_resume(self):
        rows = [row('a'), row('b')]
        base = {'finish_reason': 'direct', 'rationale_token_ids': [], 'winner_switches': 0}
        result = {'direct': {'a': {**base, 'answer': 'A'}, 'b': {**base, 'answer': 'A'}},
                  'rationale': {'a': {**base, 'answer': 'B'}, 'b': {**base, 'answer': 'B'}}}
        report = summary(rows, {'a': 'A', 'b': 'B'}, result)
        self.assertEqual(report['paired']['transitions']['C->W'], 1)
        self.assertEqual(report['paired']['transitions']['W->C'], 1)
        with tempfile.TemporaryDirectory() as path:
            args = SimpleNamespace(output_root=Path(path))
            contract = {'modes': ['direct', 'rationale']}
            h = ensure_contract(Path(path)/'manifest.json', contract)
            self.assertEqual(h, ensure_contract(Path(path)/'manifest.json', contract))
            target = output_path(args, 'direct', 'a')
            target.parent.mkdir()
            target.write_text(json.dumps({'sample_id': 'a', 'mode': 'direct', 'answer': 'A', 'contract_sha256': h}))
            self.assertEqual(set(load_done(args, 'direct', rows, h)), {'a'})
            with self.assertRaises(RuntimeError):
                load_done(args, 'direct', rows, 'wrong')


if __name__ == '__main__':
    unittest.main()

"""Ordinary greedy CoT with the frozen PCED run's formatting/cache protocol.

The PCED implementation is imported, not modified: its source hashes remain
valid for already completed results. No gold labels enter the decoder.
"""
from __future__ import annotations

import re
import torch
from medrag.generation.question_only_pced import BatchedPCED, RATIONALE_SYSTEM, compact_cache

ARMS = ('no_rag_cot', 'base_rag_cot')


def render_prompt(tokenizer, row, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    user = 'Question:\n' + row['question'] + '\n\nOptions:\n'
    user += '\n'.join(f"{key}. {row['options'][key]}" for key in 'ABCD')
    if arm == 'base_rag_cot':
        evidence = []
        for i, doc in enumerate(row['documents'], 1):
            title = doc.get('title', '').strip()
            evidence.append(f'[{i}] ' + ((title+'\n') if title else '') + doc['text'].strip())
        user += '\n\nEvidence:\n' + '\n\n'.join(evidence)
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': RATIONALE_SYSTEM}, {'role': 'user', 'content': user}],
        tokenize=False, add_generation_prompt=True)


class BatchedCoT(BatchedPCED):
    """Reuse PCED prefill/advance, but one stream and ordinary argmax per question."""

    @torch.inference_mode()
    def generate_plain(self, rows, arm):
        logits, cache, mask = self.prefill(
            [render_prompt(self.tokenizer, row, arm) for row in rows], use_cache=True)
        states = [{'row': row, 'tokens': [], 'phase': 'rationale', 'suffix': [],
                   'finish': None, 'forced_tokens': 0} for row in rows]
        complete = []
        while states:
            token = logits.argmax(-1).tolist()
            choices = logits[:, self.labels]
            answer = choices.argmax(-1).tolist()
            keep, next_tokens = [], []
            for i, state in enumerate(states):
                if state['phase'] == 'answer':
                    row = state['row']
                    complete.append({
                        'sample_id': row['sample_id'], 'dataset': row['dataset'], 'arm': arm,
                        'answer': 'ABCD'[answer[i]], 'answer_token_id': self.labels[answer[i]],
                        'choice_scores': choices[i].tolist(),
                        'document_ids': [d['corpus_id'] for d in row['documents']] if arm == 'base_rag_cot' else [],
                        'rationale': self.tokenizer.decode(state['tokens'], skip_special_tokens=True),
                        'rationale_token_ids': state['tokens'], 'finish_reason': state['finish'],
                        'forced_marker_tokens': state['forced_tokens']})
                    continue
                if state['phase'] == 'rationale' and (token[i] in self.eos or len(state['tokens']) >= self.budget):
                    state['finish'] = 'eos' if token[i] in self.eos else 'token_budget'
                    state['suffix'], state['phase'] = list(self.suffix), 'suffix'
                if state['phase'] == 'suffix':
                    selected = state['suffix'].pop(0)
                    state['forced_tokens'] += 1
                    if not state['suffix']:
                        state['phase'] = 'answer'
                else:
                    selected = token[i]
                    state['tokens'].append(selected)
                    tail = self.tokenizer.decode(state['tokens'][-24:], skip_special_tokens=False)
                    if re.search(r'(?:^|\n)\s*Final answer:\s*$', tail, re.IGNORECASE):
                        state['phase'], state['finish'] = 'answer', 'natural_marker'
                keep.append(i)
                next_tokens.append(selected)
            if not keep:
                break
            if len(keep) != len(states):
                idx = torch.tensor(keep, device=self.device)
                cache = compact_cache(cache, idx)
                mask = mask[idx]
                states = [states[i] for i in keep]
            logits, cache, mask = self.advance(next_tokens, 1, cache, mask)
        return complete

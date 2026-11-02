"""Restore only the output-instruction block; keep the task/evidence policy."""
import re

from medrag.generation import question_focused_cot as focused
from medrag.generation.question_only_pced import BatchedPCED, RATIONALE_SYSTEM

OUTPUT_INSTRUCTION = (
    "Explain your reasoning briefly, then finish on a new line "
    "with 'Final answer: <option letter>'.")
assert RATIONALE_SYSTEM.endswith(OUTPUT_INSTRUCTION)


def restore_instruction(system):
    prefix, marker, _ = system.partition('Give a brief explanation')
    if not marker:
        raise ValueError('Expected the frozen focused output instruction')
    return prefix + OUTPUT_INSTRUCTION


NO_RAG_SYSTEM = restore_instruction(focused.NO_RAG_SYSTEM)
RAG_SYSTEM = restore_instruction(focused.RAG_SYSTEM)


def baseline_prompt(tokenizer, row, arm):
    return focused.replace_system(focused.old_baseline_prompt(tokenizer, row, arm),
        NO_RAG_SYSTEM if arm == 'no_rag_cot' else RAG_SYSTEM)


def pced_prompt(tokenizer, row, document):
    return focused.replace_system(focused.old_pced_prompt(tokenizer, row, document, 'rationale'),
        NO_RAG_SYSTEM if document is None else RAG_SYSTEM)


class RestoredCoT(focused.QuestionFocusedCoT):
    def prefill(self, prompts, **kwargs):
        system = NO_RAG_SYSTEM if self.arm == 'no_rag_cot' else RAG_SYSTEM
        return BatchedPCED.prefill(self, [focused.replace_system(p, system) for p in prompts], **kwargs)


class RestoredPCED(focused.QuestionFocusedPCED):
    def prefill(self, prompts, **kwargs):
        if len(prompts) % 9:
            raise ValueError('Expected question-major nine-stream prompts')
        prompts = [focused.replace_system(p, NO_RAG_SYSTEM if i % 9 == 0 else RAG_SYSTEM)
                   for i, p in enumerate(prompts)]
        return BatchedPCED.prefill(self, prompts, **kwargs)


def behavior(records, final_token):
    values = list(records.values())
    first = sum(bool(v['rationale_token_ids']) and v['rationale_token_ids'][0] == final_token for v in values)
    empty = sum(not re.sub(r'(?:^|\n)\s*Final answer:\s*$', '', v['rationale'], flags=re.I).strip() for v in values)
    return {'n': len(values), 'first_final_count': first, 'first_final_rate': first/len(values),
            'no_explanation_count': empty, 'no_explanation_rate': empty/len(values),
            'mean_generated_tokens': sum(len(v['rationale_token_ids']) for v in values)/len(values)}


def behavior_change(before, after, final_token):
    import numpy as np
    if set(before) != set(after) or not before:
        raise ValueError('Paired behavior cohorts must match and be nonempty')
    def is_final(v):
        return bool(v['rationale_token_ids']) and v['rationale_token_ids'][0] == final_token
    counts = np.zeros(4, dtype=int)
    for sid in before:
        counts[2*is_final(before[sid])+is_final(after[sid])] += 1
    n = len(before)
    # Multinomial resampling is exactly the paired empirical bootstrap for
    # these four binary outcomes; it never treats the two runs independently.
    draws = np.random.default_rng(42).multinomial(n, counts/n, size=5000)
    ci = np.quantile((draws[:, 1]-draws[:, 2])/n, [.025, .975]).tolist()
    return {'n': n, 'before': behavior(before, final_token), 'after': behavior(after, final_token),
            'first_final_rate_delta': float((counts[1]-counts[2])/n),
            'first_final_rate_delta_95_ci': ci,
            'first_final_to_other': int(counts[2]), 'other_to_first_final': int(counts[1])}

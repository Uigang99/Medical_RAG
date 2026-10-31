"""Short task-focused system prompts; reuse immutable baseline/PCED decoders."""
from medrag.generation.question_only_pced import BatchedPCED, RATIONALE_SYSTEM, render_prompt as old_pced_prompt
from medrag.generation.question_only_cot import BatchedCoT, render_prompt as old_baseline_prompt

NO_RAG_SYSTEM = (
    "Solve the question as stated.\n\n"
    "The question defines the task and the given facts.\n\n"
    "Use the given facts together with your knowledge. "
    "Give a brief explanation of why your answer follows.\n\n"
    "Finish on a new line with 'Final answer: <option letter>'.")
RAG_SYSTEM = (
    "Solve the question as stated.\n\n"
    "The question defines the task and the given facts. Retrieved passages "
    "provide reference knowledge; they are not an extension of the question.\n\n"
    "Use relevant information from the passages together with your knowledge. "
    "Give a brief explanation of why your answer follows, rather than a "
    "summary of the passages.\n\n"
    "Finish on a new line with 'Final answer: <option letter>'.")


def replace_system(prompt, system):
    # Only replace the first, known system message; preserve user/source text,
    # even if a retrieved passage happens to contain the old instruction.
    head, sep, tail = prompt.partition(RATIONALE_SYSTEM)
    if not sep:
        raise ValueError('Expected the frozen CoT system instruction')
    return head + system + tail


def baseline_prompt(tokenizer, row, arm):
    return replace_system(old_baseline_prompt(tokenizer, row, arm),
                          NO_RAG_SYSTEM if arm == 'no_rag_cot' else RAG_SYSTEM)


def pced_prompt(tokenizer, row, document):
    return replace_system(old_pced_prompt(tokenizer, row, document, 'rationale'),
                          NO_RAG_SYSTEM if document is None else RAG_SYSTEM)


class QuestionFocusedCoT(BatchedCoT):
    def __init__(self, *args, arm, **kwargs):
        if arm not in ('no_rag_cot', 'base_rag_cot'):
            raise ValueError(arm)
        super().__init__(*args, **kwargs)
        self.arm = arm

    def generate_plain(self, rows, arm):
        if arm != self.arm:
            raise ValueError('Wrong prompt/arm pairing')
        return super().generate_plain(rows, arm)

    def prefill(self, prompts, **kwargs):
        system = NO_RAG_SYSTEM if self.arm == 'no_rag_cot' else RAG_SYSTEM
        return super().prefill([replace_system(p, system) for p in prompts], **kwargs)


class QuestionFocusedPCED(BatchedPCED):
    def generate(self, rows, mode='rationale'):
        if mode != 'rationale' or any(len(r['documents']) != 8 for r in rows):
            raise ValueError('Only CoT with one amateur + eight experts is supported')
        return super().generate(rows, mode)

    def prefill(self, prompts, **kwargs):
        if len(prompts) % 9:
            raise ValueError('Expected question-major nine-stream prompts')
        return super().prefill([replace_system(p, NO_RAG_SYSTEM if i % 9 == 0 else RAG_SYSTEM)
                                for i,p in enumerate(prompts)], **kwargs)

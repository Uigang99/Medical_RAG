from __future__ import annotations

from .core import BenchmarkSample, PromptRequest, RetrievedDocument


SYSTEM_PROMPT = "You are a medical QA assistant. Follow the user prompt strictly and output only the final answer."
QWEN_MODE = "/no_think"

DEFAULT_OPEN_ENDED_INSTRUCTION = (
    "Produce only the final biomedical answer in the reference-answer style of the dataset. "
    "Match the expected answer length and format before adding explanation."
)

OPEN_ENDED_NO_CONTEXT_INSTRUCTIONS = {
    "pubmedqa": (
        "Return exactly one lowercase token: yes, no, or maybe. The entire answer must be only that token. "
        "Do not output rationale, explanation, punctuation, extra words, XML tags, or thinking tags."
    ),
    "bioasq": (
        'BioASQ references are often short expert answers. If the question asks "Is/Are/Does/Do/Can/Has/Have", '
        "start with yes or no and add at most one brief qualifier only when needed. If the question asks "
        '"Which/What/Name/List", output only the exact biomedical entity, gene, protein, disease, drug, organism, '
        "or a comma-separated compact list. Do not turn factoid or list answers into explanatory paragraphs. "
        "Use 1-12 words for factoid/list answers and 1-3 sentences only for summary questions that clearly "
        "require explanation."
    ),
    "covidqa": (
        "COVID-QA references are compact scientific answers. Target 15-35 words. Prefer the exact number, date, "
        "entity, mechanism, method, observation, or finding asked for. Use paper-specific wording when the "
        "question asks about a study; avoid unrelated general pandemic facts."
    ),
    "mashqa": (
        "MASH-QA references are consumer-health answers. Target 40-90 words. Start with a direct patient-readable "
        "answer, then add practical context or caveats only when they help answer the question. Include symptoms, "
        "causes, tests, treatments, risks, or care steps only if the question asks for them."
    ),
}

OPEN_ENDED_CONTEXT_INSTRUCTIONS = {
    "pubmedqa": (
        "Use the retrieved abstracts exactly like the other RAG datasets, but answer in the PubMedQA label format. "
        "Return exactly one lowercase token: yes, no, or maybe. Infer the label from the abstract authors final "
        "conclusion, not from whether the evidence is perfect. Use this priority order. First, choose no for clear "
        "negative or null conclusions: no effect, no association, no benefit, no difference, no reduction, no "
        "predictive value, not useful, not safe, not reliable, not sufficient, not necessary, does not explain, "
        "does not reduce, cannot be relied upon, or the main endpoint is negative even if a subgroup is positive. "
        "Negative/null conclusions are no, not maybe. Second, choose yes for supportive conclusions: useful, "
        "effective, safe, feasible, reliable, predictive, associated, involved, implicated, beneficial, "
        "recommended, prevents, reduces, increases, improves, changes management, detects, predicts, explains, "
        "should be considered, supports the hypothesis, appears to, seems to, or may be implicated. Cautious "
        "wording can still be yes when the conclusion has a supportive direction. Third, choose maybe only when "
        "the conclusion itself is genuinely mixed, unresolved, conditional, explicitly uncertain, says more "
        "evidence is needed before deciding, or gives no overall yes/no answer. If unsure between no and maybe "
        "for a null finding, choose no. If unsure between yes and maybe for a supportive conclusion, choose yes. "
        "The entire answer must be only that token."
    ),
    "bioasq": (
        "BioASQ answers must match the expected answer type as tightly as possible. If the question is yes/no, "
        "output exactly yes or no unless a qualifier is absolutely required by the question. If the question asks "
        "Which/What/Name/List, output only the exact biomedical entity, gene, protein, disease, drug, organism, "
        "number, or a compact comma-separated list. Do not write a full sentence for factoid, list, entity, "
        "numeric, or yes/no answers. Use 1-8 words whenever possible. Use 1-2 concise sentences only when the "
        "question explicitly asks for an explanation, mechanism, role, association, or summary. Do not add "
        "background, citations, or context recap."
    ),
    "covidqa": (
        "COVID-QA answers should be as short as the reference answer permits. If the question asks for a number, "
        "percentage, date, entity, method, mutation, location, tissue, organism, or short phrase, output only that "
        "exact answer phrase without a full sentence. If the question asks what/which/who/where/when/how many, "
        "prefer 1-10 words. If the question asks why/how/mechanism/finding/definition, answer in one compact "
        "sentence, usually 10-25 words. Do not add general COVID background, extra examples, or explanatory padding."
    ),
    "mashqa": (
        "MASH-QA answers are consumer-health answers and should usually be more complete than BioASQ/COVID-QA. "
        "Start with a direct patient-readable answer, then include the main practical details, caveats, or care "
        "steps supported by the context. For broad questions about symptoms, causes, treatments, risks, prevention, "
        "lifestyle changes, or care instructions, target 60-120 words. For narrow factual questions, answer briefly "
        "but still in plain consumer-health language. Avoid unsupported medical advice and do not add unrelated details."
    ),
}

def _context_block(docs: list[RetrievedDocument], max_doc_chars: int) -> str:
    if not docs:
        return ""
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        header = (
            f"[{idx}] source={doc.source} db_id={doc.db_id} "
            f"corpus_id={doc.corpus_id} doc_id={doc.doc_id}"
        )
        blocks.append(f"{header}\n{doc.text_for_context(max_doc_chars)}")
    return "\n\n".join(blocks)


def _dataset_instruction(sample: BenchmarkSample, has_context: bool) -> str:
    dataset = sample.dataset.lower()
    instructions = OPEN_ENDED_CONTEXT_INSTRUCTIONS if has_context else OPEN_ENDED_NO_CONTEXT_INSTRUCTIONS
    return instructions.get(dataset, DEFAULT_OPEN_ENDED_INSTRUCTION)


def _build_mcq_user_prompt(sample: BenchmarkSample, docs: list[RetrievedDocument], max_doc_chars: int) -> str:
    option_text = "\n".join(sample.option_lines())
    context = _context_block(docs, max_doc_chars=max_doc_chars)
    sections = [
        QWEN_MODE,
        "You are a medical expert. Answer the following medical multiple-choice question by selecting the single best option.",
        "Output exactly one uppercase option letter and nothing else.",
        "Do not output the option text, explanation, punctuation, markdown, confidence, or reasoning.",
        "Valid option letters:",
        ", ".join(sorted(sample.options or {})),
    ]
    if context:
        sections.extend(
            [
                "Retrieved Context:",
                context,
                "Use the retrieved context only when it is relevant. Ignore off-topic or misleading snippets.",
            ]
        )
    sections.extend(["Question:", sample.question, "Options:", option_text, "Answer:"])
    return "\n\n".join(section for section in sections if section).strip()


def _build_open_ended_user_prompt(sample: BenchmarkSample, docs: list[RetrievedDocument], max_doc_chars: int) -> str:
    context = _context_block(docs, max_doc_chars=max_doc_chars)
    has_context = bool(context)
    dataset_instruction = _dataset_instruction(sample, has_context=has_context)

    if has_context:
        global_rules = [
            "Dataset-specific answer style overrides all global rules. If it asks for only a token, entity, number, phrase, or compact list, do not write a full sentence.",
            "The first sentence or phrase must directly answer the question.",
            "Always answer the question itself. Retrieved context is optional supporting evidence, not a restriction on whether you may answer.",
            "Use retrieved context when it directly supports the asked proposition or provides canonical details for the expected answer style.",
            "Ignore retrieved snippets that are off-topic, overly narrow, contradictory, or likely to distort the answer.",
            "If the retrieved context is insufficient, missing the answer, or not useful, silently ignore it and answer from your biomedical knowledge in the dataset's expected style.",
            "Never say that the provided context does not contain, mention, list, identify, specify, or provide the answer.",
            "The final answer must not include the words context, retrieved, provided, information found, or based strictly.",
            "Never refuse, hedge, or give a meta-answer about context availability; the output must be the best direct answer to the question.",
            "Match the typical reference-answer style, length, terminology, and explanatory depth of this dataset.",
            "Preserve medically important scope, negation, uncertainty, temporality, and qualifiers.",
            "If the question asks for a list, mechanism, symptom set, causes, uses, or risks, include the central expected items.",
            "Do not mention retrieved context, source quality, reasoning process, confidence, prompt instructions, or evaluation.",
            "Do not output markdown, headings, bullets, numbering, code fences, role labels, XML tags, or thinking tags.",
            "Never output <think>, </think>, analysis, reasoning traces, or multiple candidate answers.",
            "For PubMedQA, ignore all other instructions and output only one lowercase label.",
        ]
    else:
        global_rules = [
            "The first sentence must directly answer the question.",
            "Match the typical reference-answer style, length, terminology, and explanatory depth of this dataset.",
            "Preserve medically important scope, negation, uncertainty, temporality, and qualifiers.",
            "If the question asks for a list, mechanism, symptom set, causes, uses, or risks, include the central expected items.",
            "Do not mention reasoning process, confidence, prompt instructions, or evaluation.",
            "Do not output markdown, headings, bullets, numbering, code fences, role labels, XML tags, or thinking tags.",
            "Never output <think>, </think>, analysis, reasoning traces, or multiple candidate answers.",
            "For PubMedQA, ignore all other instructions and output only one lowercase label.",
        ]

    sections = [
        QWEN_MODE,
        "You are a medical open-ended QA assistant.",
        f"Dataset: {sample.dataset}",
        "Dataset-specific answer style:",
        dataset_instruction,
        "Global answer rules:",
        "\n".join(f"- {rule}" for rule in global_rules),
    ]
    if has_context:
        sections.extend(["Retrieved Context:", context])
    sections.extend(["Question:", sample.question, "Answer:"])
    return "\n\n".join(section for section in sections if section).strip()


def build_prompt_request(
    sample: BenchmarkSample,
    case_id: str,
    docs: list[RetrievedDocument],
    max_doc_chars: int,
) -> PromptRequest:
    if sample.task == "mcq":
        user_content = _build_mcq_user_prompt(sample, docs, max_doc_chars=max_doc_chars)
    else:
        user_content = _build_open_ended_user_prompt(sample, docs, max_doc_chars=max_doc_chars)

    return PromptRequest(
        sample_id=sample.id,
        case_id=case_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

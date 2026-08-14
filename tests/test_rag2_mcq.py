from __future__ import annotations

import math
import re
import unittest
from types import SimpleNamespace

from medrag.rag2_generation import (
    build_teacher_forced_request,
    flatten_generation_stats,
    generation_stats,
    score_teacher_forced_output,
    teacher_forced_target_encoding,
)
from medrag.rag2_mcq import (
    DECISIVE_REPAIR_RATIONALE_GUIDANCE,
    DOCUMENT_PROMPT_VERSION,
    FIGURE4_COMPACT_RATIONALE_GUIDANCE,
    PAPER_RATIONALE_INSTRUCTION,
    PAPER_EXACT_RATIONALE_INSTRUCTION,
    PAPER_EXACT_TERMINAL_FORMAT_INSTRUCTION,
    PROMPT_VERSION,
    RETRIEVED_EVIDENCE_INSTRUCTION,
    build_document_messages,
    build_documents_messages,
    build_no_rag_messages,
    build_paper_answer_format_document_messages,
    build_paper_exact_documents_messages,
    build_paper_exact_no_rag_messages,
    build_paper_exact_terminal_documents_messages,
    build_paper_exact_terminal_no_rag_messages,
    append_paper_exact_terminal_answer,
    paper_exact_terminal_regex,
    parse_mcq_output,
    parse_paper_exact_mcq_output,
    parse_paper_exact_terminal_mcq_output,
    render_paper_document_view,
)
from medrag.rag2_supporting_evidence import (
    INLINE_SUPPORTING_TRACE_PROMPT_VERSION,
    INLINE_SUPPORTING_TRACE_TERMINAL_REGEX,
    build_inline_supporting_trace_messages,
    build_supporting_sentence_messages,
    parse_supporting_sentence_output,
    render_tagged_document_view,
    segment_document_view,
    split_inline_supporting_trace,
)


SAMPLE = {
    "question": "Which vitamin is supplied primarily by animal sources?",
    "options": {
        "A": "Vitamin B12",
        "B": "Vitamin C",
        "C": "Vitamin B7",
        "D": "Vitamin D",
    },
    "answer": "A",
}


class CharacterTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[object]]:
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def encode(self, text: str, **_: object) -> list[int]:
        return [ord(character) for character in text]


class Rag2PaperPromptTests(unittest.TestCase):
    def test_paper_exact_terminal_adds_only_the_serialization_sentence(self) -> None:
        exact = build_paper_exact_no_rag_messages(SAMPLE)[0]["content"]
        terminal = build_paper_exact_terminal_no_rag_messages(SAMPLE)[0]["content"]
        self.assertEqual(
            terminal,
            exact.replace(
                PAPER_EXACT_RATIONALE_INSTRUCTION,
                f"{PAPER_EXACT_RATIONALE_INSTRUCTION} {PAPER_EXACT_TERMINAL_FORMAT_INSTRUCTION}",
                1,
            ),
        )

    def test_paper_exact_terminal_documents_are_raw_text_only(self) -> None:
        content = build_paper_exact_terminal_documents_messages(
            SAMPLE,
            [{"source": "pmc", "stable_id": "pmc:1", "title": "Noise title", "text": "Evidence one."},
             {"source": "cpg", "stable_id": "cpg:2", "text": "Evidence two."}],
        )[0]["content"]
        self.assertIn("Documents:\nEvidence one.\n\nEvidence two.", content)
        self.assertNotIn("pmc:1", content)
        self.assertNotIn("Noise title", content)

    def test_paper_exact_terminal_parser_requires_exact_last_line(self) -> None:
        text = (
            "Vitamin B12 is supplied primarily by animal-derived foods.\n"
            "Therefore, the answer is (A) Vitamin B12."
        )
        parsed = parse_paper_exact_terminal_mcq_output(text, SAMPLE["options"])
        self.assertEqual(parsed.final_answer, "A")
        self.assertEqual(parsed.rationale_query, text)
        self.assertFalse(parsed.parse_errors)
        malformed = parse_paper_exact_terminal_mcq_output(
            "Vitamin B12 is supplied primarily by animal foods. Therefore, the answer is (A) Vitamin B12.",
            SAMPLE["options"],
        )
        self.assertIsNone(malformed.final_answer)

    def test_terminal_repair_preserves_reasoning_and_appends_canonical_line(self) -> None:
        repaired = append_paper_exact_terminal_answer(
            "Reasoning remains.\nTherefore, the answer is A",
            SAMPLE["options"],
            "A",
        )
        self.assertEqual(
            repaired,
            "Reasoning remains.\nTherefore, the answer is (A) Vitamin B12.",
        )
        regex = paper_exact_terminal_regex(SAMPLE["options"])
        self.assertRegex(repaired, re.compile(regex))

    def test_paper_exact_prompt_has_no_terminal_answer_contract(self) -> None:
        content = build_paper_exact_no_rag_messages(SAMPLE)[0]["content"]
        self.assertEqual(
            content,
            f"{PAPER_EXACT_RATIONALE_INSTRUCTION}\nHere is the question: "
            "Which vitamin is supplied primarily by animal sources?\n"
            "A. Vitamin B12\nB. Vitamin C\nC. Vitamin B7\nD. Vitamin D",
        )
        self.assertNotIn("Therefore, the answer is", content)
        self.assertNotIn("concise", content.casefold())

    def test_paper_exact_document_prompt_only_appends_document_text(self) -> None:
        content = build_paper_exact_documents_messages(
            SAMPLE,
            [{"source": "pmc", "stable_id": "pmc:1", "title": "Title", "text": "Body"}],
        )[0]["content"]
        self.assertIn(PAPER_EXACT_RATIONALE_INSTRUCTION, content)
        self.assertIn("Documents:\nBody", content)
        self.assertNotIn("pmc:1", content)
        self.assertNotIn("Title", content)
        self.assertNotIn("[1]", content)
        self.assertNotIn("ignore the document", content.casefold())

    def test_paper_exact_parser_accepts_only_a_final_explicit_option(self) -> None:
        parsed = parse_paper_exact_mcq_output("B12 is chiefly from animal products.\nFinal answer: A", SAMPLE["options"])
        self.assertEqual(parsed.final_answer, "A")
        self.assertFalse(parsed.parse_errors)
        unparsed = parse_paper_exact_mcq_output("B12 is chiefly from animal products.", SAMPLE["options"])
        self.assertIsNone(unparsed.final_answer)

    def test_paper_exact_parser_accepts_markdown_and_decision_cues_without_rewriting(self) -> None:
        parsed = parse_paper_exact_mcq_output(
            "The clinically relevant source is animal-derived food.\n"
            "**Conclusion:** The correct answer is **A. Vitamin B12**."
            " This is consistent with its animal-source requirement.",
            SAMPLE["options"],
        )
        self.assertEqual(parsed.final_answer, "A")
        self.assertIn("**Conclusion:**", parsed.visible_text)

        decision = parse_paper_exact_mcq_output(
            "Therefore, the most appropriate option is vitamin B12 (option A).",
            SAMPLE["options"],
        )
        self.assertEqual(decision.final_answer, "A")

    def test_paper_exact_parser_recovers_terminal_decision_sentence(self) -> None:
        parsed = parse_paper_exact_mcq_output(
            "From this information, we can conclude that the affected structure is: "
            "B. Membrane currents are generated at nodes of Ranvier. This explains the finding.",
            SAMPLE["options"],
        )
        self.assertEqual(parsed.final_answer, "B")
        self.assertFalse(parsed.parse_errors)

        text_only = parse_paper_exact_mcq_output(
            "Based on the available information, the most appropriate choice is "
            "Vitamin C, because it is the explicitly selected option in this example.",
            SAMPLE["options"],
        )
        self.assertEqual(text_only.final_answer, "B")
        self.assertFalse(text_only.parse_errors)

    def test_paper_exact_parser_does_not_turn_none_of_above_into_option_a(self) -> None:
        parsed = parse_paper_exact_mcq_output(
            "The correct answer is: None of the above (A, B, C, or D).",
            SAMPLE["options"],
        )
        self.assertIsNone(parsed.final_answer)
        self.assertIn("missing_paper_answer_conclusion", parsed.parse_errors)

    def test_paper_exact_parser_rejects_explicit_multiple_options(self) -> None:
        parsed = parse_paper_exact_mcq_output(
            "The correct answer is A and B.",
            SAMPLE["options"],
        )
        self.assertIsNone(parsed.final_answer)
        self.assertIn("ambiguous_multiple_final_answers", parsed.parse_errors)

    def test_no_rag_prompt_uses_paper_instruction_with_compact_figure4_guidance(self) -> None:
        messages = build_no_rag_messages(SAMPLE)

        self.assertEqual(PROMPT_VERSION, "rag2_mcq_rationale_paper_focused_v4")
        self.assertEqual([message["role"] for message in messages], ["user"])
        content = messages[0]["content"]
        self.assertIn(PAPER_RATIONALE_INSTRUCTION, content)
        self.assertIn(FIGURE4_COMPACT_RATIONALE_GUIDANCE, content)
        self.assertIn("one focused, coherent paragraph", content)
        self.assertIn("decisive medical evidence supporting the selected answer", content)
        self.assertIn("briefly contrast another option", content)
        self.assertIn("Do not review the options one by one", content)
        self.assertNotIn("step-by-step", content)
        self.assertIn("Here is the question:", content)
        self.assertIn("Therefore, the answer is", content)
        self.assertNotIn("50-90 words", content)
        self.assertNotIn("never exceed 120 words", content)
        self.assertNotIn("words", FIGURE4_COMPACT_RATIONALE_GUIDANCE)
        self.assertNotIn("sentences", FIGURE4_COMPACT_RATIONALE_GUIDANCE)
        self.assertNotIn(RETRIEVED_EVIDENCE_INSTRUCTION, content)

    def test_compact_retry_requests_rewrite_without_increasing_the_length_contract(self) -> None:
        content = build_no_rag_messages(SAMPLE, compact_retry=True)[0]["content"]

        self.assertIn("previous response exceeded the available output space", content)
        self.assertIn("Do not compare alternatives", content)
        self.assertIn(DECISIVE_REPAIR_RATIONALE_GUIDANCE, content)
        self.assertNotIn(FIGURE4_COMPACT_RATIONALE_GUIDANCE, content)
        self.assertNotIn("more tokens", content)

    def test_choice_anchored_compact_retry_does_not_reconsider_the_selected_option(self) -> None:
        content = build_no_rag_messages(
            SAMPLE,
            format_retry=True,
            selected_answer="A",
            compact_retry=True,
        )[0]["content"]

        self.assertIn("fixed the answer as (A) Vitamin B12", content)
        self.assertIn("Do not reconsider or change that option", content)
        self.assertIn("without discussing alternatives, ambiguity, or flaws", content)

    def test_single_document_prompt_adds_only_neutral_evidence_instruction(self) -> None:
        messages = build_document_messages(
            SAMPLE,
            {"source": "textbooks", "stable_id": "doc-1", "text": "Vitamin B12 occurs in animal foods."},
        )

        self.assertEqual(DOCUMENT_PROMPT_VERSION, "rag2_mcq_document_rationale_paper_focused_v3")
        content = messages[0]["content"]
        self.assertIn(PAPER_RATIONALE_INSTRUCTION, content)
        self.assertIn(FIGURE4_COMPACT_RATIONALE_GUIDANCE, content)
        self.assertIn(RETRIEVED_EVIDENCE_INSTRUCTION, content)
        self.assertIn("provided below as additional context", content)
        self.assertIn("medical reasoning that supports the selected answer", content)
        self.assertNotIn("combining the evidence with your own medical knowledge", content)
        self.assertNotIn("Integrate document-derived information", content)
        self.assertNotIn("must use", content.lower())
        self.assertIn("Retrieved document:", content)
        self.assertIn("Therefore, the answer is", content)
        self.assertNotIn("document is relevant", content)
        self.assertNotIn("document is correct", content)
        self.assertNotIn("Ignore off-topic or misleading documents", content)

    def test_multi_document_prompt_uses_the_same_neutral_evidence_instruction(self) -> None:
        messages = build_documents_messages(
            SAMPLE,
            [{"source": "textbooks", "stable_id": "doc-1", "text": "Vitamin B12 occurs in animal foods."}],
        )

        content = messages[0]["content"]
        self.assertIn(PAPER_RATIONALE_INSTRUCTION, content)
        self.assertIn(FIGURE4_COMPACT_RATIONALE_GUIDANCE, content)
        self.assertIn(RETRIEVED_EVIDENCE_INSTRUCTION, content)
        self.assertIn("Retrieved documents:", content)
        self.assertNotIn("Ignore off-topic or misleading documents", content)

    def test_parser_accepts_paper_style_response_without_rationale_marker(self) -> None:
        output = (
            "Vitamin B12 is naturally concentrated in animal-derived foods, unlike the other listed vitamins. "
            "Therefore, the answer is (A) Vitamin B12."
        )

        parsed = parse_mcq_output(output, SAMPLE["options"])

        self.assertEqual(parsed.final_answer, "A")
        self.assertEqual(parsed.parse_errors, [])
        self.assertEqual(parsed.rationale_query, output)
        self.assertEqual(
            parsed.rationale_only,
            "Vitamin B12 is naturally concentrated in animal-derived foods, unlike the other listed vitamins.",
        )
        self.assertEqual(parsed.answer_conclusion, "Therefore, the answer is (A) Vitamin B12.")
        self.assertEqual(output[slice(*parsed.rationale_only_span)], parsed.rationale_only)
        self.assertEqual(output[slice(*parsed.answer_conclusion_span)], parsed.answer_conclusion)

    def test_one_generation_produces_both_rationale_ppl_scopes(self) -> None:
        output = (
            "Vitamin B12 is supplied primarily by animal foods. "
            "Therefore, the answer is (A) Vitamin B12."
        )
        parsed = parse_mcq_output(output, SAMPLE["options"])
        token_ids = [ord(character) for character in output]
        token_logprobs = []
        for index, token_id in enumerate(token_ids):
            in_reasoning = parsed.rationale_only_span[0] <= index < parsed.rationale_only_span[1]
            logprob = -math.log(2.0 if in_reasoning else 4.0)
            token_logprobs.append({token_id: logprob})
        generated = SimpleNamespace(
            token_ids=token_ids,
            logprobs=token_logprobs,
            cumulative_logprob=sum(next(iter(row.values())) for row in token_logprobs),
        )

        stats = flatten_generation_stats(
            generation_stats(generated, CharacterTokenizer(), output, SAMPLE["options"])
        )

        self.assertAlmostEqual(stats["rationale_only_ppl"], 2.0)
        self.assertAlmostEqual(stats["answer_conclusion_ppl"], 4.0)
        self.assertEqual(stats["rationale_with_answer_ppl"], stats["rationale_ppl"])
        self.assertGreater(stats["rationale_with_answer_ppl"], 2.0)
        self.assertLess(stats["rationale_with_answer_ppl"], 4.0)

    def test_inline_support_suffix_is_excluded_from_rationale_and_answer_ppl(self) -> None:
        response = (
            "Vitamin B12 is supplied primarily by animal foods.\n"
            "Therefore, the answer is (A) Vitamin B12."
        )
        raw = (
            f"{response}\n<SUPPORTING_SENTENCES>\n"
            '{"supporting_sentence_ids":[],"support_links":[],"none_reason":"No document premise was used."}\n'
            "</SUPPORTING_SENTENCES>"
        )
        parsed = parse_mcq_output(response, SAMPLE["options"])
        token_ids = [ord(character) for character in raw]
        token_logprobs = []
        for index, token_id in enumerate(token_ids):
            in_reasoning = parsed.rationale_only_span[0] <= index < parsed.rationale_only_span[1]
            in_answer = parsed.answer_conclusion_span[0] <= index < parsed.answer_conclusion_span[1]
            logprob = -math.log(2.0 if in_reasoning else 4.0 if in_answer else 9.0)
            token_logprobs.append({token_id: logprob})
        generated = SimpleNamespace(
            token_ids=token_ids,
            logprobs=token_logprobs,
            cumulative_logprob=sum(next(iter(row.values())) for row in token_logprobs),
        )

        stats = flatten_generation_stats(
            generation_stats(
                generated,
                CharacterTokenizer(),
                raw,
                SAMPLE["options"],
                parsed_output=parsed,
            )
        )

        self.assertAlmostEqual(stats["rationale_only_ppl"], 2.0)
        self.assertAlmostEqual(stats["answer_conclusion_ppl"], 4.0)

    def test_teacher_forcing_scores_the_fixed_target_suffix_only(self) -> None:
        target_text = (
            "Vitamin B12 is supplied primarily by animal foods. "
            "Therefore, the answer is (A) Vitamin B12."
        )
        tokenizer = CharacterTokenizer()
        target = teacher_forced_target_encoding(tokenizer, target_text, SAMPLE["options"])
        request = build_teacher_forced_request(tokenizer, "PROMPT", target, max_model_len=4096)
        prompt_logprobs = [
            {token_id: -math.log(7.0) if index < request["prompt_token_count"] else -math.log(2.0)}
            for index, token_id in enumerate(request["input_ids"])
        ]
        generated = SimpleNamespace(
            prompt_token_ids=request["input_ids"],
            prompt_logprobs=prompt_logprobs,
        )

        stats = score_teacher_forced_output(generated, request)

        self.assertAlmostEqual(stats["rationale_with_answer"]["ppl"], 2.0)
        self.assertAlmostEqual(stats["rationale_only"]["ppl"], 2.0)
        self.assertAlmostEqual(stats["answer_conclusion"]["ppl"], 2.0)

    def test_truncated_unmarked_response_reports_missing_conclusion(self) -> None:
        output = "The obstruction causes hydronephrosis and progressive compression of the renal parenchyma."

        parsed = parse_mcq_output(output, SAMPLE["options"])

        self.assertEqual(parsed.rationale, output)
        self.assertIsNone(parsed.final_answer)
        self.assertIn("missing_paper_answer_conclusion", parsed.parse_errors)
        self.assertNotIn("missing_rationale_marker", parsed.parse_errors)

    def test_supporting_sentence_offsets_match_exact_document_prompt_view(self) -> None:
        document = {
            "title": "Vitamin source",
            "text": "Vitamin B12 occurs in animal foods. It supports red cell production.",
        }
        document_view = render_paper_document_view(document, max_doc_chars=2600)
        prompt = build_paper_answer_format_document_messages(
            SAMPLE,
            document,
            max_doc_chars=2600,
        )[0]["content"]
        sentences = segment_document_view(document_view, document_title=document["title"])

        self.assertTrue(prompt.endswith(f"Document:\n{document_view}"))
        self.assertEqual([sentence["sentence_id"] for sentence in sentences], ["S001", "S002"])
        self.assertNotIn(document["title"], [sentence["text"] for sentence in sentences])
        for sentence in sentences:
            self.assertEqual(
                document_view[sentence["char_start"] : sentence["char_end"]],
                sentence["text"],
            )

    def test_supporting_sentence_parser_maps_ids_to_exact_text(self) -> None:
        document_view = "Vitamin B12 occurs in animal foods. It supports red cell production."
        sentences = segment_document_view(document_view)
        parsed = parse_supporting_sentence_output(
            '{"supporting_sentence_ids":["S001","S002","S002"]}',
            sentences,
        )

        self.assertTrue(parsed["quality_pass"])
        self.assertEqual(parsed["sentence_ids"], ["S001", "S002"])
        self.assertEqual(parsed["sentences"][0]["text"], "Vitamin B12 occurs in animal foods.")

    def test_supporting_sentence_parser_extracts_embedded_json_and_normalizes_leading_zero_ids(self) -> None:
        sentences = segment_document_view(
            "Vitamin B12 occurs in animal foods. It supports red cell production."
        )
        parsed = parse_supporting_sentence_output(
            'Here is the result:\n{"supporting_sentence_ids":["S0001", "S0002"]}',
            sentences,
        )

        self.assertTrue(parsed["quality_pass"])
        self.assertFalse(parsed["retry_required"])
        self.assertEqual(parsed["parse_mode"], "embedded_json")
        self.assertEqual(parsed["sentence_ids"], ["S001", "S002"])

    def test_supporting_sentence_parser_requires_json_when_only_ids_are_emitted(self) -> None:
        sentences = segment_document_view("Vitamin B12 occurs in animal foods.")
        parsed = parse_supporting_sentence_output("S001", sentences)

        self.assertFalse(parsed["quality_pass"])
        self.assertTrue(parsed["retry_required"])
        self.assertIn("supporting_sentence_json_required", parsed["quality_issues"])

    def test_support_prompt_shows_title_without_assigning_an_id(self) -> None:
        document_view = "Vitamin source\nVitamin B12 occurs in animal foods."
        sentences = segment_document_view(document_view, document_title="Vitamin source")
        messages = build_supporting_sentence_messages(
            SAMPLE,
            document_view,
            sentences,
            "Vitamin B12 is derived from animal foods. Therefore, the answer is (A) Vitamin B12.",
            document_title="Vitamin source",
        )

        content = messages[0]["content"]
        self.assertIn("Document title (metadata only; it has no sentence ID):\nVitamin source", content)
        self.assertIn("[S001] Vitamin B12 occurs in animal foods.", content)
        self.assertNotIn("[S001] Vitamin source", content)

    def test_support_prompt_is_separate_from_answer_generation_contract(self) -> None:
        document_view = "Vitamin B12 occurs in animal foods."
        sentences = segment_document_view(document_view)
        messages = build_supporting_sentence_messages(
            SAMPLE,
            document_view,
            sentences,
            "Vitamin B12 is derived from animal foods. Therefore, the answer is (A) Vitamin B12.",
        )

        content = messages[0]["content"]
        self.assertIn('"supporting_sentence_ids"', content)
        self.assertIn("[S001] Vitamin B12 occurs in animal foods.", content)
        self.assertIn("Do not solve the question again", content)

    def test_inline_support_trace_keeps_attribution_in_the_same_response_schema(self) -> None:
        document_view = "Vitamin source\nVitamin B12 occurs in animal foods. It supports red cell production."
        sentences = segment_document_view(document_view, document_title="Vitamin source")
        messages = build_inline_supporting_trace_messages(
            SAMPLE,
            document_view,
            sentences,
            document_title="Vitamin source",
        )
        content = messages[0]["content"]

        self.assertEqual(INLINE_SUPPORTING_TRACE_PROMPT_VERSION, "rag2_llama3_inline_rationale_answer_support_v6")
        self.assertIn("SUPPORTING_SENTENCE_JSON:", content)
        self.assertIn("For NONE, output exactly", content)
        self.assertIn("topic, word, disease-name, entity, or option overlap alone", content)
        self.assertIn("recall-oriented attribution", content)
        self.assertIn("include it rather than omitting it", content)
        self.assertIn('"supporting_sentence_ids"', content)
        self.assertNotIn("rationale_quote", content)
        self.assertIn("[S001] Vitamin B12 occurs in animal foods.", content)
        self.assertNotIn("[S001] Vitamin source", content)
        self.assertEqual(
            render_tagged_document_view(document_view, sentences, document_title="Vitamin source"),
            "Vitamin source\n[S001] Vitamin B12 occurs in animal foods.\n[S002] It supports red cell production.",
        )

    def test_inline_support_trace_parses_sentence_ids_and_separates_answer_prefix(self) -> None:
        sentences = segment_document_view("Vitamin B12 occurs in animal foods. It supports red cell production.")
        raw = (
            "Vitamin B12 is supplied primarily by animal foods.\n"
            "Therefore, the answer is (A) Vitamin B12.\n"
            'SUPPORTING_SENTENCE_JSON: {"supporting_sentence_ids":["S001"]}'
        )
        split = split_inline_supporting_trace(raw)
        parsed = parse_supporting_sentence_output(
            split["support_raw_generation"],
            sentences,
            require_json_only=True,
        )

        self.assertEqual(split["quality_issues"], [])
        self.assertTrue(split["response_text"].endswith("Therefore, the answer is (A) Vitamin B12."))
        self.assertTrue(parsed["quality_pass"])
        self.assertEqual(parsed["sentence_ids"], ["S001"])
        self.assertEqual(parsed["support_links"], [])

    def test_inline_support_trace_accepts_none_and_rejects_unknown_sentence_ids(self) -> None:
        sentences = segment_document_view("Vitamin B12 occurs in animal foods.")
        none = parse_supporting_sentence_output(
            '{"supporting_sentence_ids":"NONE"}',
            sentences,
            require_json_only=True,
        )
        unknown = parse_supporting_sentence_output(
            '{"supporting_sentence_ids":["S999"]}',
            sentences,
            require_json_only=True,
        )

        self.assertTrue(none["quality_pass"])
        self.assertEqual(none["sentence_ids"], [])
        self.assertFalse(unknown["quality_pass"])
        self.assertIn("unknown_supporting_sentence_ids", unknown["quality_issues"])

    def test_inline_support_trace_accepts_list_encoded_none_and_inline_terminal_header(self) -> None:
        sentences = segment_document_view("Vitamin B12 occurs in animal foods.")
        raw = (
            "Vitamin B12 is supplied primarily by animal foods. "
            "Therefore, the answer is (A) Vitamin B12. "
            'SUPPORTING_SENTENCE_JSON: {"supporting_sentence_ids":["NONE"]}'
        )
        split = split_inline_supporting_trace(raw)
        parsed = parse_supporting_sentence_output(
            split["support_raw_generation"],
            sentences,
            require_json_only=True,
        )

        self.assertEqual(split["response_text"], "Vitamin B12 is supplied primarily by animal foods. Therefore, the answer is (A) Vitamin B12.")
        self.assertTrue(parsed["quality_pass"])
        self.assertEqual(parsed["sentence_ids"], [])

    def test_inline_terminal_regex_allows_free_reasoning_and_only_canonical_support_suffix(self) -> None:
        valid = (
            "Vitamin B12 is supplied primarily by animal foods.\n"
            "Therefore, the answer is (A) Vitamin B12.\n"
            'SUPPORTING_SENTENCE_JSON: {"supporting_sentence_ids":["S001","S002"]}'
        )
        invalid = valid + "\nextra text"

        self.assertIsNotNone(re.fullmatch(INLINE_SUPPORTING_TRACE_TERMINAL_REGEX, valid))
        self.assertIsNone(re.fullmatch(INLINE_SUPPORTING_TRACE_TERMINAL_REGEX, invalid))


if __name__ == "__main__":
    unittest.main()

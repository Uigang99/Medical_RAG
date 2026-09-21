from __future__ import annotations

import json
import unittest

from scripts.extract_rag2_top8_standalone_propositions import (
    EXPECTED_SYSTEM_PROMPT_SHA256,
    SYSTEM_PROMPT,
    document_prompt,
    parse_model_output,
    proposition_schema,
    sha256_text,
    validate_prompt_contract,
)


class StandalonePropositionExtractionTest(unittest.TestCase):
    def test_prompt_contains_only_source_passage_and_text(self) -> None:
        document = {
            "source": "pubmed",
            "title": "A source title",
            "text": "Aspirin reduced events. Aspirin increased bleeding.",
        }
        prompt = document_prompt(document)
        self.assertEqual(prompt, "SOURCE PASSAGE:\nAspirin reduced events. Aspirin increased bleeding.")
        for forbidden in ("pubmed", "A source title", "gold_answer", "semantic_label", "Answer choices"):
            self.assertNotIn(forbidden, prompt)

    def test_parser_preserves_claim_and_exact_evidence(self) -> None:
        raw = json.dumps(
            {
                "propositions": [
                    {
                        "claim": "Aspirin reduced events in the study population.",
                        "evidence": "Aspirin reduced events in the study population.",
                    }
                ],
            }
        )
        parsed, errors = parse_model_output(raw, "Aspirin reduced events in the study population.")
        self.assertEqual(errors, [])
        assert parsed is not None
        self.assertTrue(parsed["propositions"][0]["evidence_exact_match"])

    def test_parser_flags_nonexact_evidence_without_dropping_claim(self) -> None:
        raw = json.dumps(
            {
                "propositions": [
                    {
                        "claim": "An unsupported claim.",
                        "evidence": "Not an exact source span.",
                    }
                ],
            }
        )
        parsed, errors = parse_model_output(raw, "Only one sentence.")
        assert parsed is not None
        self.assertIn("proposition_0_evidence_not_exact", errors)
        self.assertEqual(len(parsed["propositions"]), 1)

    def test_empty_output_is_preserved_and_flagged(self) -> None:
        parsed, errors = parse_model_output('{"propositions":[]}', "A title only.")
        self.assertIn("empty_propositions", errors)
        self.assertEqual(parsed["propositions"], [])

    def test_schema_matches_v2_claim_evidence_contract(self) -> None:
        schema = proposition_schema()
        item = schema["properties"]["propositions"]["items"]
        self.assertEqual(item["required"], ["claim", "evidence"])

    def test_exact_recovered_prompt_hash(self) -> None:
        self.assertEqual(sha256_text(SYSTEM_PROMPT), EXPECTED_SYSTEM_PROMPT_SHA256)
        validate_prompt_contract()


if __name__ == "__main__":
    unittest.main()

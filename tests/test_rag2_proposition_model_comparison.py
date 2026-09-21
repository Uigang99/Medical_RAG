import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_rag2_proposition_models as pilot


class ComparisonTests(unittest.TestCase):
    def test_original_prompt_unchanged(self):
        self.assertEqual(pilot.sha(pilot.PROMPTS['standalone_v2']),pilot.EXPECTED_SYSTEM_PROMPT_SHA256)

    def test_envelope_not_content_repaired(self):
        text='{"propositions":[{"claim":"A reduces B.","evidence":"A reduces B."}]}'
        for suffix in ['<turn|>','<|im_end|>','<eos>','']:
            parsed,errors=pilot.parse_model_output(pilot.clean_completion(text+suffix),'A reduces B.')
            self.assertEqual(errors,[])
            self.assertEqual(len(parsed['propositions']),1)
        parsed,errors=pilot.parse_model_output(pilot.clean_completion(text[:-2]),'A reduces B.')
        self.assertIsNone(parsed)
        self.assertIn('invalid_json',errors)

    def test_no_invented_semantic_quality(self):
        self.assertIn('Do not guess missing words',pilot.FAITHFUL_PROMPT)
        self.assertIn('Repeat these details',pilot.FAITHFUL_PROMPT)


if __name__=='__main__':unittest.main()

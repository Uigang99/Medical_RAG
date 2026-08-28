from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_rag2_codex_semantic_filter_inputs import (  # noqa: E402
    SEMANTIC_FOUR_LABELS,
    SEMANTIC_LABELS,
    target_from_semantic_label,
)
from train_rag2_filter_model_paper import (  # noqa: E402
    SEMANTIC_FOUR_LABEL_NAMES,
    SEMANTIC_FOUR_LABEL_TOKENS,
    normalize_training_label,
    training_label_spec,
)


class SemanticMulticlassFilterTests(unittest.TestCase):
    def test_materializer_preserves_four_coherent_semantic_targets(self) -> None:
        for label in SEMANTIC_FOUR_LABELS:
            self.assertEqual(target_from_semantic_label(label, "semantic_four"), (label, label))
        self.assertIsNone(target_from_semantic_label("indeterminate_or_mixed", "semantic_four"))

    def test_binary_materialization_remains_backward_compatible(self) -> None:
        self.assertEqual(target_from_semantic_label("direct_support", "binary"), ("helpful", "Helpful"))
        self.assertEqual(
            target_from_semantic_label("supporting_evidence", "binary"),
            ("helpful", "Helpful"),
        )
        for label in ("no_evidence", "misleading_evidence", "indeterminate_or_mixed"):
            self.assertEqual(target_from_semantic_label(label, "binary"), ("not helpful", "Not Helpful"))

    def test_trainer_exposes_atomic_target_contract_for_four_classes(self) -> None:
        names, tokens = training_label_spec("semantic_four")
        self.assertEqual(names, SEMANTIC_FOUR_LABEL_NAMES)
        self.assertEqual(tokens, SEMANTIC_FOUR_LABEL_TOKENS)
        self.assertEqual(len(set(tokens)), 4)

    def test_semantic_label_normalization_accepts_spaces_and_underscores(self) -> None:
        for label in SEMANTIC_LABELS:
            self.assertEqual(normalize_training_label(label), label)
            self.assertEqual(normalize_training_label(label.replace("_", " ")), label)


if __name__ == "__main__":
    unittest.main()

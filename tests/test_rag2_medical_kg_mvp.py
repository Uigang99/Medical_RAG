from __future__ import annotations

import unittest

from medrag.kg.biomedical_mvp import (
    aggregate_edge_assertions,
    category_from_tuis,
    infer_relation,
)


class BiomedicalKgMvpTest(unittest.TestCase):
    def test_umls_type_grouping(self) -> None:
        self.assertEqual(category_from_tuis(["T047"]), "DiseaseOrDisorder")
        self.assertEqual(category_from_tuis(["T121"]), "DrugOrChemical")
        self.assertEqual(category_from_tuis(["T116"]), "GeneOrProtein")
        self.assertEqual(category_from_tuis(["T999"]), "OtherBiomedical")

    def test_generic_subject_does_not_create_causal_fact(self) -> None:
        generic = {"entity_id": "UMLS:C1", "category": "OtherBiomedical"}
        disease = {"entity_id": "UMLS:C2", "category": "DiseaseOrDisorder"}
        relation = infer_relation(
            generic,
            disease,
            sentence_text="Formation can lead to hemolysis.",
            between_text=" can lead to ",
        )
        self.assertIsNone(relation)

    def test_treatment_orientation_is_semantic_not_surface_order(self) -> None:
        disease = {"entity_id": "UMLS:C1", "category": "DiseaseOrDisorder"}
        drug = {"entity_id": "UMLS:C2", "category": "DrugOrChemical"}
        relation = infer_relation(
            disease,
            drug,
            sentence_text="Pneumonia is treated with amoxicillin.",
            between_text=" is treated with ",
        )
        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertEqual((relation.subject_id, relation.predicate, relation.object_id), ("UMLS:C2", "TREATS", "UMLS:C1"))

    def test_negated_assertion_is_not_merged_with_affirmed(self) -> None:
        rows = [
            {
                "subject_id": "UMLS:C1", "predicate": "CAUSES", "object_id": "UMLS:C2",
                "polarity": polarity, "certainty": "asserted", "chunk_id": chunk,
                "source": "pmc", "rule_id": "causal_between_v1", "rule_score": 0.84,
            }
            for polarity, chunk in (("affirmed", "a"), ("negated", "b"))
        ]
        edges = aggregate_edge_assertions(rows)
        self.assertEqual(len(edges), 2)
        self.assertEqual({row["polarity"] for row in edges}, {"affirmed", "negated"})


if __name__ == "__main__":
    unittest.main()

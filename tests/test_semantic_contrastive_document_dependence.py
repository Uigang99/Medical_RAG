import unittest

from scripts.evaluate_rag2_semantic_contrastive_document_dependence import (
    select_highest_rank,
    source_matched_derangement,
)


def row(sample, stable, rank, length, source="pubmed", label="direct_support"):
    return {
        "sample_id": sample,
        "document_stable_id": stable,
        "document_source": source,
        "document_text": "x" * length,
        "semantic_label": label,
        "rerank_rank": rank,
        "pair_id": f"{sample}:{stable}",
    }


class DocumentDependenceTest(unittest.TestCase):
    def test_selects_highest_rerank_document(self):
        values = [row("q", "a", 4, 10), row("q", "b", 1, 10), row("q", "c", 0, 10, label="no_evidence")]
        self.assertEqual(select_highest_rank(values, "direct_support")["document_stable_id"], "b")

    def test_derangement_preserves_source_and_changes_question(self):
        records = [
            {"sample_id": f"q{i}", "direct": row(f"q{i}", f"d{i}", i, 100 + i, source="pubmed")}
            for i in range(5)
        ] + [
            {"sample_id": f"c{i}", "direct": row(f"c{i}", f"e{i}", i, 200 + i, source="cpg")}
            for i in range(2)
        ]
        mapping = source_matched_derangement(records)
        self.assertEqual(set(mapping), {value["sample_id"] for value in records})
        for value in records:
            donor = mapping[value["sample_id"]]
            self.assertNotEqual(donor["sample_id"], value["sample_id"])
            self.assertEqual(donor["direct"]["document_source"], value["direct"]["document_source"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from medrag import proposition_pair_candidates as c
from scripts import prepare_rag2_common_pair_candidates as runner


class QuietProgress:
    def stage(self, *args, **kwargs): pass
    def update(self, *args): pass
    def complete(self, *args): pass


def settings(**overrides):
    return dict(shared_entity_k=1, direct_kg_k=1, semantic_k=2, document_adjacent_k=2,
                min_cosine=.7, max_common_fraction=.01, reserve_cross_document=True,
                audit_per_stratum=2, seed=7, **overrides)


def fixture():
    ps = [dict(idx=0, pid="p0", did="d0", pos=0, claim="X mechanism", entities={"x"}),
          dict(idx=1, pid="p1", did="d0", pos=1, claim="X finding", entities={"x"}),
          dict(idx=2, pid="p2", did="d1", pos=0, claim="Y result", entities={"y"}),
          dict(idx=3, pid="p3", did="d2", pos=0, claim="Another statement", entities=set())]
    es = {e: dict(eid=e, name="specific "+e, level="canonical", canonical=e, doc_frequency=1) for e in ("x", "y")}
    ts = [dict(tid="t1", idx=2, s="x", r="relates to", o="y")]
    matrix = np.eye(4, dtype=np.float32)
    return ps, es, ts, matrix


class CandidateTests(unittest.TestCase):
    def test_channels_and_local_witness(self):
        ps, es, ts, sim = fixture()
        sim[0, 3] = sim[3, 0] = .9
        r = c.select_pairs(ps, es, ts, sim, settings(), 1000, "q")
        by_pair = {(p["source"], p["target"]): p for p in r["pairs"]}
        self.assertIn("shared_entity", by_pair["p0", "p1"]["reasons"])
        self.assertIn("document_adjacent", by_pair["p0", "p1"]["reasons"])
        self.assertIn("direct_kg", by_pair["p0", "p2"]["reasons"])
        self.assertIn("semantic", by_pair["p0", "p3"]["reasons"])
        self.assertEqual(by_pair["p0", "p2"]["bridge_triples"][0]["idx"], 2)
        self.assertEqual(r, c.select_pairs(ps, es, ts, sim, settings(), 1000, "q"))

    def test_outside_top8_witness_never_enters(self):
        ps, es, ts, sim = fixture()
        ts[0]["idx"] = 999
        r = c.select_pairs(ps, es, ts, sim, settings(), 1000, "q")
        self.assertFalse(any("direct_kg" in p["reasons"] for p in r["pairs"]))

    def test_local_and_generic_identity_not_promoted(self):
        ps, es, ts, sim = fixture()
        es["x"]["level"] = "lexical_only"
        es["y"]["name"] = "patient"
        r = c.select_pairs(ps, es, ts, sim, settings(), 1000, "q")
        self.assertFalse(any("shared_entity" in p["reasons"] or "direct_kg" in p["reasons"] for p in r["pairs"]))
        self.assertTrue(any("document_adjacent" in p["reasons"] for p in r["pairs"]))

    def test_common_entity_needs_second_signal(self):
        ps, es, ts, sim = fixture()
        es["x"]["doc_frequency"] = 50
        r = c.select_pairs(ps, es, [], sim, settings(), 100, "q")
        self.assertFalse(any("shared_entity" in p["reasons"] for p in r["pairs"]))
        sim[0, 1] = sim[1, 0] = .8
        r = c.select_pairs(ps, es, [], sim, settings(), 100, "q")
        self.assertTrue(any("shared_entity" in p["reasons"] for p in r["pairs"]))

    def test_cross_document_slot_and_unilateral_union(self):
        ps, es, ts, sim = fixture()
        ps[2]["did"] = "d0"
        sim[0, 1] = sim[1, 0] = .95
        sim[0, 2] = sim[2, 0] = .94
        sim[0, 3] = sim[3, 0] = .8
        r = c.select_pairs(ps, es, [], sim, settings(), 1000, "q")
        proposals = [p for p in r["pairs"] if "p0" in p["reasons"].get("semantic", [])]
        self.assertEqual(len(proposals), 2)
        self.assertTrue(any(p["target"] == "p3" for p in proposals))

    def test_no_forced_connectivity_and_audit_is_not_negative(self):
        ps, es, ts, sim = fixture()
        for i, p in enumerate(ps):
            p["did"] = str(i)
            p["entities"] = set()
        r = c.select_pairs(ps, {}, [], sim, settings(), 1000, "q")
        self.assertEqual(r["pairs"], [])
        self.assertEqual(r["stats"]["isolated_propositions"], 4)
        self.assertTrue(all(x["status"] == "unassessed" for x in r["rejected_audit"]))
        self.assertEqual(r["audit_strata"]["cross_document"]["population"], 6)

    def test_atomic_inventory_kg_resume_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = [dict(document_id=f"d{i}", rank=i+1, propositions=[]) for i in range(8)]
            docs[0]["propositions"] = [dict(proposition_id="p0", claim="X mechanism"), dict(proposition_id="p1", claim="X finding")]
            docs[1]["propositions"] = [dict(proposition_id="p2", claim="Y result")]
            docs[2]["propositions"] = [dict(proposition_id="p3", claim="Other statement")]
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps(dict(sample_id=f"q{i}", dataset="medqa", question="SECRET_QUESTION_GOLD", documents=docs)) for i in range(2))+"\n")
            db = c.connect(root / "candidates.sqlite3")
            c.initialize(db, "test")
            s = c.ingest(db, source, 2, 4)
            self.assertEqual(s["unique_cooccurring_pairs"], 6)
            self.assertEqual(s["within_question_all_pair_occurrences"], 12)
            self.assertEqual(c.ingest(db, source, 2, 4), s)
            kg = root / "kg.sqlite3"
            with sqlite3.connect(kg) as k:
                k.executescript("""
                  CREATE TABLE proposition(proposition_id,document_id,claim);
                  CREATE TABLE entity(entity_node_id,preferred_surface,canonical_concept_id,normalization_level);
                  CREATE TABLE mention(proposition_id,entity_node_id,assertion_node_id);
                  CREATE TABLE binary_relation(edge_id,assertion_node_id,proposition_id,subject_entity_id,relation_surface,object_entity_id);
                """)
                k.executemany("INSERT INTO proposition VALUES (?,?,?)", [("p0", "d0", "X mechanism"), ("p1", "d0", "X finding"), ("p2", "d1", "Y result")])
                k.executemany("INSERT INTO entity VALUES (?,?,?,?)", [("x", "specific x", "x", "canonical"), ("y", "specific y", "y", "canonical")])
                k.executemany("INSERT INTO mention VALUES (?,?,?)", [("p0", "x", "a0"), ("p1", "x", "a1"), ("p2", "x", "a2"), ("p2", "y", "a2")])
                k.execute("INSERT INTO binary_relation VALUES ('t1','a2','p2','x','relates to','y')")
            c.import_kg(db, kg, chunk_size=2)
            c.import_kg(db, kg, chunk_size=2)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM triple").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT kg_present FROM proposition WHERE pid='p3'").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT doc_frequency FROM entity WHERE eid='x'").fetchone()[0], 2)
            a = SimpleNamespace(output_root=root, shared_entity_k=1, direct_kg_k=1, semantic_k=2,
                                min_cosine=.7, max_common_fraction=.01, audit_per_stratum=2, seed=7)
            # Production selection path, including an identical-command cached rerun.
            vectors = np.ones((4, 3), dtype=np.float32) / np.sqrt(3)
            runner.select_all(a, db, vectors, QuietProgress())
            runner.select_all(a, db, vectors, QuietProgress())
            self.assertEqual(db.execute("SELECT COUNT(*) FROM question_result").fetchone()[0], 2)
            n = db.execute("SELECT COUNT(*) FROM pair").fetchone()[0]
            self.assertLessEqual(n, 6)
            result = runner.export(a, db, QuietProgress())
            self.assertEqual(result["unique_selected_pairs"], n)
            for name in ("propositions.jsonl", "pairs.jsonl", "question_candidates.jsonl"):
                text = (root / name).read_text()
                self.assertNotIn("SECRET_QUESTION_GOLD", text)
                for line in text.splitlines():
                    json.loads(line)
            for p in map(json.loads, (root / "pairs.jsonl").read_text().splitlines()):
                self.assertEqual(p["graphs_to_evaluate"], ["A", "B", "C"])
                self.assertEqual(p["status"], "unassessed")
            db.close()

    def test_inventory_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            row = dict(sample_id="q", dataset="medqa", documents=[dict(document_id=str(i),rank=i+1,
                propositions=[dict(proposition_id="duplicate",claim="same")]) for i in range(8)])
            source.write_text(json.dumps(row)+"\n")
            db = c.connect(root / "out.sqlite3")
            c.initialize(db, "test")
            with self.assertRaises(ValueError):
                c.ingest(db, source, 1, 8)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM proposition").fetchone()[0], 0)
            self.assertIsNone(c.get_meta(db, "inventory_done"))
            db.close()


if __name__ == "__main__":
    unittest.main()

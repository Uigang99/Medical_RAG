from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rag2_stage6_7_surface_kg_union_full.py"
SPEC = importlib.util.spec_from_file_location("stage67", SCRIPT)
assert SPEC and SPEC.loader
stage67 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage67)


def endpoint(mention: str, entity: str, text: str, level: str = "canonical", lexical: str | None = None) -> dict:
    canonical = entity if level == "canonical" else None
    return {
        "mention_id": mention,
        "entity_node_id": entity,
        "canonical_concept_id": canonical,
        "lexical_cluster_id": lexical,
        "normalization_level": level,
        "normalization_method": "test",
        "normalization_confidence": "high" if canonical else "abstain",
        "normalization_abstained_reason": "" if canonical else "test",
        "text": text,
        "start": 0,
        "end": len(text),
    }


def frame(job: int, document: str, proposition: str, subjects: list[dict], objects: list[dict]) -> dict:
    relation = "treats"
    claim = f"Subject {relation} object {job}."
    return {
        "job_index": job,
        "document_id": document,
        "proposition_id": proposition,
        "source": "test",
        "segment": "top8",
        "claim": claim,
        "assertion_id": 0,
        "assertion": {"id": 0, "assertion_type": "fact", "scope_link": "independent"},
        "frame": {
            "arity": "binary",
            "assertion_id": 0,
            "implicit_relation": False,
            "relation": {"text": relation, "start": 8, "end": 14},
            "evidence_span": {"text": claim[:-1], "start": 0, "end": len(claim) - 1},
            "subjects": subjects,
            "objects": objects,
        },
        "origin": "original",
        "decision_source": "test",
        "stage4_action": "keep",
        "stage4_reason": ["ordinary_explicit_frame"],
    }


class Stage67Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = stage67.connect_database(self.root / "graph.sqlite3", "test-contract")

    def tearDown(self) -> None:
        self.db.close()
        self.temp.cleanup()

    def test_binary_quarantine_and_hard_identity(self) -> None:
        rows = [
            frame(0, "D1", "P1", [endpoint("J0::SUBJECT0", "UMLS:C1", "Drug")],
                  [endpoint("J0::OBJECT0", "UMLS:C2", "Disease")]),
            frame(1, "D2", "P2", [endpoint("J1::SUBJECT0", "UMLS:C1", "The drug")],
                  [endpoint("J1::OBJECT0", "MENTION:M2", "Outcome", "lexical_only", "LEX:X")]),
            frame(2, "D3", "P3", [endpoint("J2::SUBJECT0", "MENTION:M3", "Agent", "lexical_only", "LEX:X")],
                  [endpoint("J2::OBJECT0", "MENTION:M4", "Outcome", "lexical_only", "LEX:Y")]),
            frame(3, "D4", "P4", [endpoint("J3::SUBJECT0", "UMLS:C9", "Alias A")],
                  [endpoint("J3::OBJECT0", "UMLS:C9", "Alias B")]),
            frame(4, "D5", "P5", [endpoint("J4::SUBJECT0", "UMLS:C10", "Combination")],
                  [endpoint("J4::OBJECT0", "UMLS:C11", "One"), endpoint("J4::OBJECT1", "UMLS:C12", "Two")]),
        ]
        stage67.insert_batch(self.db, rows, 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM assertion").fetchone()[0], 5)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM binary_relation").fetchone()[0], 3)
        reasons = dict(self.db.execute("SELECT reason,COUNT(*) FROM quarantined_assertion GROUP BY reason"))
        self.assertEqual(reasons, {"nonbinary_endpoint_count": 1, "normalized_endpoints_collapse": 1})
        self.assertEqual(self.db.execute("SELECT value FROM metadata WHERE key='processed_frames'").fetchone()[0], "5")
        self.assertEqual(self.db.execute("SELECT mention_count FROM entity WHERE entity_node_id='UMLS:C1'").fetchone()[0], 2)

    def test_resume_cursor_is_committed_with_batch(self) -> None:
        row = frame(0, "D1", "P1", [endpoint("J0::SUBJECT0", "UMLS:C1", "Drug")],
                    [endpoint("J0::OBJECT0", "UMLS:C2", "Disease")])
        stage67.insert_batch(self.db, [row], 0)
        self.assertEqual(self.db.execute("SELECT value FROM metadata WHERE key='processed_frames'").fetchone()[0], "1")
        with self.assertRaises(sqlite3.IntegrityError):
            stage67.insert_batch(self.db, [row], 1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM assertion").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT value FROM metadata WHERE key='processed_frames'").fetchone()[0], "1")

    def test_validate_frame_rejects_missing_identity(self) -> None:
        row = frame(0, "D1", "P1", [endpoint("J0::SUBJECT0", "UMLS:C1", "Drug")],
                    [endpoint("J0::OBJECT0", "UMLS:C2", "Disease")])
        del row["frame"]["objects"][0]["entity_node_id"]
        with self.assertRaises(RuntimeError):
            stage67.validate_frame(row)

    def test_preflight_manifest_refresh_requires_no_durable_output(self) -> None:
        output = self.root / "manifest_case"
        output.mkdir()
        stage67.atomic_json(output / "manifest.json", {"contract": {"old": True}, "contract_sha256": "old"})
        digest = stage67.ensure_manifest(output, {"new": True})
        self.assertEqual(json.loads((output / "manifest.json").read_text())["contract_sha256"], digest)
        (output / "graph.sqlite3").touch()
        with self.assertRaises(RuntimeError):
            stage67.ensure_manifest(output, {"newer": True})


if __name__ == "__main__":
    unittest.main()

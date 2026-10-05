#!/usr/bin/env python3
"""Materialize the frozen Stage-5 frames and audit corpus-level connectivity.

Stage 6 is a deterministic graph materialization step.  It copies the frozen
surface relation and endpoint identities into a provenance-first assertion
graph and emits a direct binary edge only for one-subject/one-object frames.

Stage 7 adds no biomedical fact.  It builds inverted indexes over the shared
canonical entity nodes selected in Stage 5, materializes cross-document bridge
diagnostics, and computes document connected components.  Lexical clusters are
kept as soft diagnostics and never become hard graph identity.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from extract_rag2_top8_standalone_propositions import TwoLineProgress  # noqa: E402


STAGE5 = ROOT / "datasets/kg/rag2_test6545_union88294_stage5_contextual_entity_generalization_v4"
OUTPUT = ROOT / "datasets/kg/rag2_test6545_union88294_stage7_surface_kg_v1"
VERSION = "rag2_test6545_union88294_stage7_surface_kg_v1"
EXPECTED = {
    "frames": 1_034_619,
    "mentions": 2_249_489,
    # Step 4 saw 550,900 propositions in 78,410 documents, but propositions
    # and documents whose every frame was deferred are intentionally absent
    # from the Stage-5 graph-ready stream.  These are the exact identities
    # present in all 1,034,619 normalized Stage-5 frames.
    "propositions": 547_496,
    "documents": 78_406,
}
EXPECTED_STAGE5_CONTRACT = "97bddb2ea4b3c859484b8c18042f8a13f5294de5f333bcc25caece17a8aacbd8"
DEFAULT_TRANSACTION = 5_000
MIN_FREE_GIB_FULL = 35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5-root", type=Path, default=STAGE5)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--transaction-size", type=int, default=DEFAULT_TRANSACTION)
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Development-only prefix limit. Zero enforces the complete Stage-5 contract.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "frames": args.stage5_root / "normalized_frames.jsonl",
        "mentions": args.stage5_root / "normalized_mentions.jsonl",
        "manifest": args.stage5_root / "manifest.json",
        "summary": args.stage5_root / "summary.json",
        "complete": args.stage5_root / "COMPLETE.json",
    }


def validate_stage5(args: argparse.Namespace) -> dict[str, Any]:
    paths = input_paths(args)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage-5 input(s): {missing}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    complete = json.loads(paths["complete"].read_text(encoding="utf-8"))
    observed_contracts = {manifest.get("contract_sha256"), complete.get("contract_sha256")}
    if observed_contracts != {EXPECTED_STAGE5_CONTRACT}:
        raise RuntimeError(f"Unexpected Stage-5 contract: {observed_contracts}")
    if int(summary.get("scope", {}).get("frames", -1)) != EXPECTED["frames"]:
        raise RuntimeError("Stage-5 frame count differs from the frozen Stage-6/7 contract")
    if int(summary.get("scope", {}).get("mentions", -1)) != EXPECTED["mentions"]:
        raise RuntimeError("Stage-5 mention count differs from the frozen Stage-6/7 contract")
    if int(summary.get("relation_or_endpoint_mutations", -1)) != 0:
        raise RuntimeError("Stage-5 reports relation or endpoint mutations")
    if int(summary.get("context_parse_errors_abstained", -1)) != 0:
        raise RuntimeError("Stage-5 reports contextual parsing failures")
    if int(summary.get("missing_entity_nodes", -1)) != 0:
        raise RuntimeError("Stage-5 reports missing entity nodes")
    if args.limit_frames < 0 or args.limit_frames > EXPECTED["frames"]:
        raise ValueError(f"--limit-frames must be in [0,{EXPECTED['frames']}]")
    if args.transaction_size < 1:
        raise ValueError("--transaction-size must be positive")
    if not args.limit_frames:
        free_gib = shutil.disk_usage(args.output_root.parent).free / 2**30
        if free_gib < MIN_FREE_GIB_FULL:
            raise RuntimeError(
                f"Only {free_gib:.1f} GiB free; full deterministic materialization requires "
                f"at least {MIN_FREE_GIB_FULL} GiB"
            )
    first = next(iter_jsonl(paths["frames"]))
    validate_frame(first)
    return {"paths": paths, "manifest": manifest, "summary": summary, "complete": complete}


def experiment_contract(args: argparse.Namespace, loaded: Mapping[str, Any]) -> dict[str, Any]:
    paths = loaded["paths"]
    target = args.limit_frames or EXPECTED["frames"]
    return {
        "version": VERSION,
        "purpose": "Materialize frozen source-grounded surface frames and index hard canonical cross-document connectivity.",
        "changed_variable": "No semantic variable changes; this is deterministic graph materialization and indexing.",
        "inputs": {
            "stage5_contract_sha256": EXPECTED_STAGE5_CONTRACT,
            "frames": file_identity(paths["frames"]),
            "mentions": file_identity(paths["mentions"]),
            **{
                name: {"path": str(paths[name].resolve()), "sha256": sha256_file(paths[name])}
                for name in ("manifest", "summary", "complete")
            },
        },
        "scope": {
            "target_frames": target,
            "full_contract": args.limit_frames == 0,
            "qa_labels_used": False,
            "answers_or_choices_used": False,
        },
        "stage6": {
            "source_of_truth": "Document--Proposition--Assertion--Mention--Entity provenance graph",
            "binary_view": "Entity--verbatim surface relation--Entity only for exactly one subject and one object",
            "predicate_normalization": False,
            "multiargument_cartesian_expansion": False,
            "self_edges_after_normalization": "quarantined as assertions, not emitted as direct binary edges",
            "relation_and_endpoint_mutation": False,
        },
        "stage7": {
            "hard_cross_document_identity": "canonical entity_node_id selected by frozen Stage 5 only",
            "lexical_cluster_policy": "diagnostic soft index only; never hard graph identity",
            "new_biomedical_relation_inference": False,
            "outputs": "canonical entity/proposition/document indexes, bridge diagnostics, and document components",
        },
        "integrity_gate": {
            "all_input_frames_accounted": True,
            "all_endpoint_mentions_have_nodes": True,
            "dangling_references_max": 0,
            "relation_surface_mutations_max": 0,
            "unsafe_multiargument_expansions_max": 0,
            "hard_noncanonical_bridges_max": 0,
        },
        "execution": {
            "transaction_size": args.transaction_size,
            "resume": "SQLite committed-frame cursor; atomic JSONL exports",
            "gpu_required": False,
        },
        "code": {
            "path": str(SCRIPT.resolve()),
            "sha256": sha256_file(SCRIPT),
        },
    }


def contract_sha(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def ensure_manifest(output: Path, contract: Mapping[str, Any]) -> str:
    value = {"contract": contract, "contract_sha256": contract_sha(contract)}
    path = output / "manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != value:
            durable_names = {
                "graph.sqlite3", "stage6_complete.json", "stage7_complete.json",
                "summary.json", "COMPLETE.json",
            }
            durable = sorted(name for name in durable_names if (output / name).exists())
            durable.extend(sorted(item.name for item in output.glob("*.jsonl")))
            if durable:
                raise RuntimeError(
                    f"Stage-6/7 resume contract mismatch with durable output {durable[:8]}; "
                    f"use a new output root: {output}"
                )
            # A preflight writes only manifest.json and .run.lock.  It is safe
            # to refresh that manifest when the exact script/input contract
            # changes because no graph row has been materialized yet.
            atomic_json(path, value)
    else:
        atomic_json(path, value)
    return value["contract_sha256"]


def validate_frame(row: Mapping[str, Any]) -> None:
    required = ("job_index", "document_id", "proposition_id", "source", "segment", "claim", "frame")
    missing = [key for key in required if key not in row]
    if missing:
        raise RuntimeError(f"Stage-5 frame missing keys {missing}")
    if row.get("stage4_action") != "keep":
        raise RuntimeError(f"Non-keep frame reached Stage 6: job={row.get('job_index')}")
    frame = row["frame"]
    relation = frame.get("relation") or {}
    if not str(relation.get("text") or "").strip():
        raise RuntimeError(f"Empty surface relation: job={row.get('job_index')}")
    for role in ("subjects", "objects"):
        if not isinstance(frame.get(role), list):
            raise RuntimeError(f"Invalid {role}: job={row.get('job_index')}")
        for endpoint in frame[role]:
            for key in ("mention_id", "entity_node_id", "normalization_level", "text"):
                if not endpoint.get(key):
                    raise RuntimeError(f"Endpoint missing {key}: job={row.get('job_index')}")


def connect_database(path: Path, contract_sha256: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS document(
          document_id TEXT PRIMARY KEY, source TEXT NOT NULL, segment TEXT NOT NULL,
          consistent INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS proposition(
          proposition_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, source TEXT NOT NULL,
          segment TEXT NOT NULL, claim TEXT NOT NULL, consistent INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS assertion(
          assertion_node_id TEXT PRIMARY KEY, job_index INTEGER NOT NULL UNIQUE,
          source_assertion_id INTEGER NOT NULL, proposition_id TEXT NOT NULL,
          document_id TEXT NOT NULL, source TEXT NOT NULL, segment TEXT NOT NULL,
          claim TEXT NOT NULL, relation_surface TEXT NOT NULL,
          relation_start INTEGER NOT NULL, relation_end INTEGER NOT NULL,
          evidence_text TEXT NOT NULL, evidence_start INTEGER NOT NULL, evidence_end INTEGER NOT NULL,
          arity TEXT NOT NULL, implicit_relation INTEGER NOT NULL,
          assertion_metadata_json TEXT NOT NULL, origin TEXT NOT NULL,
          decision_source TEXT NOT NULL, stage4_reason_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS entity(
          entity_node_id TEXT PRIMARY KEY, canonical_concept_id TEXT,
          normalization_level TEXT NOT NULL, preferred_surface TEXT NOT NULL,
          mention_count INTEGER NOT NULL, consistent INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS mention(
          mention_id TEXT PRIMARY KEY, assertion_node_id TEXT NOT NULL,
          proposition_id TEXT NOT NULL, document_id TEXT NOT NULL, source TEXT NOT NULL,
          segment TEXT NOT NULL, role TEXT NOT NULL, role_index INTEGER NOT NULL,
          surface TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
          entity_node_id TEXT NOT NULL, canonical_concept_id TEXT,
          lexical_cluster_id TEXT, normalization_level TEXT NOT NULL,
          normalization_method TEXT NOT NULL, normalization_confidence TEXT NOT NULL,
          normalization_abstained_reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS binary_relation(
          edge_id TEXT PRIMARY KEY, assertion_node_id TEXT NOT NULL,
          proposition_id TEXT NOT NULL, document_id TEXT NOT NULL, source TEXT NOT NULL,
          subject_mention_id TEXT NOT NULL, subject_entity_id TEXT NOT NULL,
          relation_surface TEXT NOT NULL,
          object_mention_id TEXT NOT NULL, object_entity_id TEXT NOT NULL,
          evidence_text TEXT NOT NULL, assertion_metadata_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS quarantined_assertion(
          assertion_node_id TEXT PRIMARY KEY, reason TEXT NOT NULL,
          subject_count INTEGER NOT NULL, object_count INTEGER NOT NULL);
        """
    )
    existing = connection.execute("SELECT value FROM metadata WHERE key='contract_sha256'").fetchone()
    if existing and existing[0] != contract_sha256:
        raise RuntimeError("SQLite cache contract mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES('contract_sha256',?)", (contract_sha256,)
    )
    connection.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('processed_frames','0')")
    connection.commit()
    return connection


def endpoint_tuple(
    row: Mapping[str, Any], assertion_node_id: str, role: str, role_index: int, endpoint: Mapping[str, Any]
) -> tuple[Any, ...]:
    return (
        str(endpoint["mention_id"]), assertion_node_id, str(row["proposition_id"]),
        str(row["document_id"]), str(row["source"]), str(row["segment"]), role, role_index,
        str(endpoint["text"]), int(endpoint.get("start", -1)), int(endpoint.get("end", -1)),
        str(endpoint["entity_node_id"]), endpoint.get("canonical_concept_id"),
        endpoint.get("lexical_cluster_id"), str(endpoint["normalization_level"]),
        str(endpoint.get("normalization_method") or ""),
        str(endpoint.get("normalization_confidence") or ""),
        str(endpoint.get("normalization_abstained_reason") or ""),
    )


def insert_batch(connection: sqlite3.Connection, rows: Sequence[Mapping[str, Any]], processed: int) -> None:
    documents: list[tuple[Any, ...]] = []
    propositions: list[tuple[Any, ...]] = []
    assertions: list[tuple[Any, ...]] = []
    mentions: list[tuple[Any, ...]] = []
    entities: list[tuple[Any, ...]] = []
    binary: list[tuple[Any, ...]] = []
    quarantined: list[tuple[Any, ...]] = []
    for row in rows:
        validate_frame(row)
        job = int(row["job_index"])
        assertion_node_id = f"ASSERT:J{job}"
        frame = row["frame"]
        relation = frame["relation"]
        relation_surface = str(relation["text"])
        evidence = frame.get("evidence_span") or row.get("assertion", {}).get("evidence_span") or {}
        assertion_meta = row.get("assertion") or {}
        documents.append((str(row["document_id"]), str(row["source"]), str(row["segment"])))
        propositions.append(
            (str(row["proposition_id"]), str(row["document_id"]), str(row["source"]),
             str(row["segment"]), str(row["claim"]))
        )
        assertions.append(
            (
                assertion_node_id, job, int(row.get("assertion_id", frame.get("assertion_id", -1))),
                str(row["proposition_id"]), str(row["document_id"]), str(row["source"]),
                str(row["segment"]), str(row["claim"]), relation_surface,
                int(relation.get("start", -1)), int(relation.get("end", -1)),
                str(evidence.get("text") or ""), int(evidence.get("start", -1)), int(evidence.get("end", -1)),
                str(frame.get("arity") or ""), int(bool(frame.get("implicit_relation"))),
                canonical_json(assertion_meta), str(row.get("origin") or ""),
                str(row.get("decision_source") or ""), canonical_json(row.get("stage4_reason") or []),
            )
        )
        subjects = frame.get("subjects") or []
        objects = frame.get("objects") or []
        for role, endpoints in (("subject", subjects), ("object", objects)):
            for role_index, endpoint in enumerate(endpoints):
                mentions.append(endpoint_tuple(row, assertion_node_id, role, role_index, endpoint))
                entities.append(
                    (
                        str(endpoint["entity_node_id"]), endpoint.get("canonical_concept_id"),
                        str(endpoint["normalization_level"]), str(endpoint["text"]),
                    )
                )
        if len(subjects) == 1 and len(objects) == 1:
            subject, obj = subjects[0], objects[0]
            if str(subject["entity_node_id"]) == str(obj["entity_node_id"]):
                quarantined.append((assertion_node_id, "normalized_endpoints_collapse", 1, 1))
            else:
                binary.append(
                    (
                        f"EDGE:{stable_hash(assertion_node_id)}", assertion_node_id,
                        str(row["proposition_id"]), str(row["document_id"]), str(row["source"]),
                        str(subject["mention_id"]), str(subject["entity_node_id"]), relation_surface,
                        str(obj["mention_id"]), str(obj["entity_node_id"]),
                        str(evidence.get("text") or ""), canonical_json(assertion_meta),
                    )
                )
        else:
            quarantined.append(
                (assertion_node_id, "nonbinary_endpoint_count", len(subjects), len(objects))
            )
    with connection:
        connection.executemany(
            """INSERT INTO document(document_id,source,segment) VALUES(?,?,?)
               ON CONFLICT(document_id) DO UPDATE SET consistent=MIN(document.consistent,
                 CASE WHEN document.source=excluded.source AND document.segment=excluded.segment THEN 1 ELSE 0 END)""",
            documents,
        )
        connection.executemany(
            """INSERT INTO proposition(proposition_id,document_id,source,segment,claim) VALUES(?,?,?,?,?)
               ON CONFLICT(proposition_id) DO UPDATE SET consistent=MIN(proposition.consistent,
                 CASE WHEN proposition.document_id=excluded.document_id AND proposition.source=excluded.source
                    AND proposition.segment=excluded.segment AND proposition.claim=excluded.claim THEN 1 ELSE 0 END)""",
            propositions,
        )
        connection.executemany(
            "INSERT INTO assertion VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", assertions
        )
        connection.executemany(
            """INSERT INTO entity(entity_node_id,canonical_concept_id,normalization_level,preferred_surface,mention_count)
               VALUES(?,?,?,?,1) ON CONFLICT(entity_node_id) DO UPDATE SET
               mention_count=entity.mention_count+1,
               consistent=MIN(entity.consistent,CASE WHEN
                 COALESCE(entity.canonical_concept_id,'')=COALESCE(excluded.canonical_concept_id,'')
                 AND entity.normalization_level=excluded.normalization_level THEN 1 ELSE 0 END)""",
            entities,
        )
        connection.executemany(
            "INSERT INTO mention VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", mentions
        )
        connection.executemany(
            "INSERT INTO binary_relation VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", binary
        )
        connection.executemany(
            "INSERT INTO quarantined_assertion VALUES(?,?,?,?)", quarantined
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='processed_frames'", (str(processed + len(rows)),)
        )


def materialize(args: argparse.Namespace, connection: sqlite3.Connection, progress: TwoLineProgress) -> None:
    target = args.limit_frames or EXPECTED["frames"]
    processed = int(connection.execute("SELECT value FROM metadata WHERE key='processed_frames'").fetchone()[0])
    if processed > target:
        raise RuntimeError(f"SQLite has {processed:,} frames but target is {target:,}")
    progress.start(2, target, initial=processed, unit="frame", estimate=90.0)
    started = time.time()
    batch: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(input_paths(args)["frames"])):
        if index < processed:
            continue
        if index >= target:
            break
        batch.append(row)
        if len(batch) >= args.transaction_size:
            insert_batch(connection, batch, index + 1 - len(batch))
            progress.done = index + 1
            elapsed = max(1e-9, time.time() - started)
            fresh = progress.done - processed
            if fresh:
                progress.estimates[1] = elapsed * target / fresh
            progress.render()
            batch.clear()
    if batch:
        current = int(connection.execute("SELECT value FROM metadata WHERE key='processed_frames'").fetchone()[0])
        insert_batch(connection, batch, current)
        progress.done = target
        progress.render()
    final = int(connection.execute("SELECT value FROM metadata WHERE key='processed_frames'").fetchone()[0])
    if final != target:
        raise RuntimeError(f"Stage-6 input ended early: processed={final:,}, target={target:,}")
    progress.complete(f"durable SQLite prefix={final:,}; database={args.output_root / 'graph.sqlite3'}")


def create_indexes(connection: sqlite3.Connection) -> None:
    with connection:
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_proposition_document ON proposition(document_id);
            CREATE INDEX IF NOT EXISTS idx_assertion_proposition ON assertion(proposition_id);
            CREATE INDEX IF NOT EXISTS idx_assertion_document ON assertion(document_id);
            CREATE INDEX IF NOT EXISTS idx_mention_assertion ON mention(assertion_node_id);
            CREATE INDEX IF NOT EXISTS idx_mention_entity ON mention(entity_node_id);
            CREATE INDEX IF NOT EXISTS idx_mention_proposition ON mention(proposition_id);
            CREATE INDEX IF NOT EXISTS idx_mention_document ON mention(document_id);
            CREATE INDEX IF NOT EXISTS idx_mention_lexical ON mention(lexical_cluster_id);
            CREATE INDEX IF NOT EXISTS idx_binary_subject ON binary_relation(subject_entity_id);
            CREATE INDEX IF NOT EXISTS idx_binary_object ON binary_relation(object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_binary_assertion ON binary_relation(assertion_node_id);
            """
        )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def rows_from_cursor(cursor: sqlite3.Cursor) -> Iterator[dict[str, Any]]:
    columns = [item[0] for item in cursor.description]
    for values in cursor:
        yield dict(zip(columns, values))


def assertion_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cursor = connection.execute("SELECT * FROM assertion ORDER BY job_index")
    for row in rows_from_cursor(cursor):
        row["assertion_metadata"] = json.loads(row.pop("assertion_metadata_json"))
        row["stage4_reason"] = json.loads(row.pop("stage4_reason_json"))
        yield row


def binary_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cursor = connection.execute(
        "SELECT * FROM binary_relation ORDER BY assertion_node_id"
    )
    for row in rows_from_cursor(cursor):
        row["assertion_metadata"] = json.loads(row.pop("assertion_metadata_json"))
        yield row


def structural_rows(connection: sqlite3.Connection) -> Iterator[dict[str, str]]:
    queries = (
        ("SELECT proposition_id,document_id FROM proposition ORDER BY proposition_id", "HAS_PROPOSITION", True),
        ("SELECT assertion_node_id,proposition_id FROM assertion ORDER BY job_index", "HAS_ASSERTION", True),
        ("SELECT mention_id,assertion_node_id,role FROM mention ORDER BY assertion_node_id,role,role_index", "", True),
        ("SELECT mention_id,entity_node_id FROM mention ORDER BY mention_id", "NORMALIZED_TO", False),
    )
    for query, relation, reverse in queries:
        cursor = connection.execute(query)
        columns = [item[0] for item in cursor.description]
        for values in cursor:
            row = dict(zip(columns, values))
            if "role" in row:
                source, target = row["assertion_node_id"], row["mention_id"]
                rel = f"HAS_{str(row['role']).upper()}_MENTION"
            elif "proposition_id" in row and "document_id" in row:
                source, target = row["document_id"], row["proposition_id"]
                rel = relation
            elif "assertion_node_id" in row:
                source, target = row["proposition_id"], row["assertion_node_id"]
                rel = relation
            else:
                source, target = row["mention_id"], row["entity_node_id"]
                rel = relation
            edge_key = f"{source}|{rel}|{target}"
            yield {"edge_id": f"STRUCT:{stable_hash(edge_key)}", "source_id": source, "relation": rel, "target_id": target}


def export_stage6(args: argparse.Namespace, connection: sqlite3.Connection, contract_sha256: str, progress: TwoLineProgress) -> dict[str, int]:
    marker = args.output_root / "stage6_complete.json"
    names = (
        "documents.jsonl", "propositions.jsonl", "assertions.jsonl", "entity_mentions.jsonl",
        "entities.jsonl", "binary_relation_edges.jsonl", "quarantined_assertions.jsonl", "structural_edges.jsonl",
    )
    if marker.is_file() and all((args.output_root / name).is_file() for name in names):
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("contract_sha256") != contract_sha256:
            raise RuntimeError("Stage-6 completion marker contract mismatch")
        progress.start(3, len(names), initial=len(names), unit="artifact", estimate=0.0)
        progress.complete("all Stage-6 graph artifacts cached")
        return {key: int(value[key]) for key in value if key not in {"contract_sha256"}}
    progress.start(3, len(names), unit="artifact", estimate=150.0)
    create_indexes(connection)
    outputs: list[tuple[str, Iterable[Mapping[str, Any]]]] = [
        ("documents.jsonl", rows_from_cursor(connection.execute("SELECT document_id,source,segment FROM document ORDER BY document_id"))),
        ("propositions.jsonl", rows_from_cursor(connection.execute("SELECT proposition_id,document_id,source,segment,claim FROM proposition ORDER BY proposition_id"))),
        ("assertions.jsonl", assertion_rows(connection)),
        ("entity_mentions.jsonl", rows_from_cursor(connection.execute("SELECT * FROM mention ORDER BY assertion_node_id,role,role_index"))),
        ("entities.jsonl", rows_from_cursor(connection.execute("SELECT entity_node_id,canonical_concept_id,normalization_level,preferred_surface,mention_count FROM entity ORDER BY entity_node_id"))),
        ("binary_relation_edges.jsonl", binary_rows(connection)),
        ("quarantined_assertions.jsonl", rows_from_cursor(connection.execute("SELECT * FROM quarantined_assertion ORDER BY assertion_node_id"))),
        ("structural_edges.jsonl", structural_rows(connection)),
    ]
    counts: dict[str, int] = {}
    for name, rows in outputs:
        counts[name.removesuffix(".jsonl")] = write_jsonl(args.output_root / name, rows)
        progress.update(1)
    value = {"contract_sha256": contract_sha256, **counts}
    atomic_json(marker, value)
    progress.complete("; ".join(f"{key}={value:,}" for key, value in counts.items()))
    return counts


class UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def grouped_canonical_mentions(connection: sqlite3.Connection) -> Iterator[tuple[str, list[tuple[str, str, str, str]]]]:
    cursor = connection.execute(
        """SELECT entity_node_id,assertion_node_id,proposition_id,document_id,source
           FROM mention WHERE canonical_concept_id IS NOT NULL
           ORDER BY entity_node_id,document_id,proposition_id,assertion_node_id"""
    )
    current: str | None = None
    values: list[tuple[str, str, str, str]] = []
    for entity_id, assertion_id, proposition_id, document_id, source in cursor:
        if current is not None and entity_id != current:
            yield current, values
            values = []
        current = entity_id
        values.append((assertion_id, proposition_id, document_id, source))
    if current is not None:
        yield current, values


def build_stage7(args: argparse.Namespace, connection: sqlite3.Connection, contract_sha256: str, progress: TwoLineProgress) -> dict[str, Any]:
    marker = args.output_root / "stage7_complete.json"
    required = ("canonical_entity_bridges.jsonl", "lexical_bridge_candidates.jsonl", "document_components.jsonl")
    if marker.is_file() and all((args.output_root / name).is_file() for name in required):
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("contract_sha256") != contract_sha256:
            raise RuntimeError("Stage-7 completion marker contract mismatch")
        progress.start(4, 3, initial=3, unit="operation", estimate=0.0)
        progress.complete("all Stage-7 connectivity artifacts cached")
        return value["metrics"]
    progress.start(4, 3, unit="operation", estimate=90.0)
    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS canonical_entity_document AS
              SELECT DISTINCT entity_node_id,document_id FROM mention
              WHERE canonical_concept_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_entity_document
              ON canonical_entity_document(entity_node_id,document_id);
            CREATE INDEX IF NOT EXISTS idx_canonical_document
              ON canonical_entity_document(document_id);
            CREATE TABLE IF NOT EXISTS canonical_entity_proposition AS
              SELECT DISTINCT entity_node_id,proposition_id FROM mention
              WHERE canonical_concept_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_entity_proposition
              ON canonical_entity_proposition(entity_node_id,proposition_id);
            CREATE INDEX IF NOT EXISTS idx_canonical_proposition
              ON canonical_entity_proposition(proposition_id);
            """
        )
    progress.update(1)
    documents = [row[0] for row in connection.execute("SELECT document_id FROM document ORDER BY document_id")]
    source_by_document = dict(connection.execute("SELECT document_id,source FROM document"))
    union = UnionFind(documents)
    bridge_path = args.output_root / "canonical_entity_bridges.jsonl"
    bridge_tmp = bridge_path.with_name(f".{bridge_path.name}.{os.getpid()}.tmp")
    bridge_counts = Counter()
    top_hubs: list[dict[str, Any]] = []
    with bridge_tmp.open("w", encoding="utf-8") as handle:
        for entity_id, occurrences in grouped_canonical_mentions(connection):
            assertions = {row[0] for row in occurrences}
            propositions = {row[1] for row in occurrences}
            document_ids = sorted({row[2] for row in occurrences})
            sources = sorted({row[3] for row in occurrences})
            bridge_counts["canonical_entities"] += 1
            bridge_counts["canonical_occurrences"] += len(occurrences)
            if len(propositions) <= 1:
                continue
            bridge_counts["cross_proposition_entities"] += 1
            if len(document_ids) > 1:
                bridge_counts["cross_document_entities"] += 1
                for other in document_ids[1:]:
                    union.union(document_ids[0], other)
            if len(sources) > 1:
                bridge_counts["cross_source_entities"] += 1
            row = {
                "entity_node_id": entity_id,
                "mention_count": len(occurrences),
                "assertion_count": len(assertions),
                "proposition_count": len(propositions),
                "document_count": len(document_ids),
                "source_count": len(sources),
                "sources": sources,
                "sample_assertion_ids": sorted(assertions)[:20],
                "sample_proposition_ids": sorted(propositions)[:20],
                "sample_document_ids": document_ids[:20],
                "hard_identity": True,
            }
            handle.write(canonical_json(row) + "\n")
            top_hubs.append(row)
    os.replace(bridge_tmp, bridge_path)
    top_hubs = sorted(top_hubs, key=lambda row: (-row["document_count"], row["entity_node_id"]))[:100]
    write_jsonl(args.output_root / "top_canonical_hubs.jsonl", top_hubs)
    progress.update(1)

    lexical_path = args.output_root / "lexical_bridge_candidates.jsonl"
    lexical_cursor = connection.execute(
        """SELECT lexical_cluster_id,COUNT(*) AS mention_count,
                  COUNT(DISTINCT proposition_id) AS proposition_count,
                  COUNT(DISTINCT document_id) AS document_count,
                  COUNT(DISTINCT source) AS source_count
           FROM mention
           WHERE normalization_level='lexical_only' AND lexical_cluster_id IS NOT NULL
           GROUP BY lexical_cluster_id HAVING COUNT(DISTINCT proposition_id)>1
           ORDER BY lexical_cluster_id"""
    )
    lexical_count = write_jsonl(
        lexical_path,
        (
            {
                "lexical_cluster_id": row[0], "mention_count": row[1],
                "proposition_count": row[2], "document_count": row[3],
                "source_count": row[4], "hard_identity": False,
            }
            for row in lexical_cursor
        ),
    )
    components: dict[str, list[str]] = defaultdict(list)
    for document_id in documents:
        components[union.find(document_id)].append(document_id)
    component_rows = []
    for members in components.values():
        members.sort()
        digest = hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:20]
        source_counts = Counter(source_by_document[document_id] for document_id in members)
        component_rows.append(
            {
                "component_id": f"COMPONENT:{digest}", "document_count": len(members),
                "document_ids": members, "source_counts": dict(sorted(source_counts.items())),
            }
        )
    component_rows.sort(key=lambda row: (-row["document_count"], row["component_id"]))
    write_jsonl(args.output_root / "document_components.jsonl", component_rows)
    metrics = {
        **dict(bridge_counts),
        "lexical_bridge_candidate_clusters": lexical_count,
        "document_components": len(component_rows),
        "nontrivial_document_components": sum(row["document_count"] > 1 for row in component_rows),
        "documents_in_nontrivial_components": sum(
            row["document_count"] for row in component_rows if row["document_count"] > 1
        ),
        "largest_document_component": max((row["document_count"] for row in component_rows), default=0),
    }
    value = {"contract_sha256": contract_sha256, "metrics": metrics}
    atomic_json(marker, value)
    progress.update(1)
    progress.complete(
        f"hard_cross_document_entities={metrics.get('cross_document_entities',0):,}; "
        f"components={metrics['document_components']:,}; largest={metrics['largest_document_component']:,}; "
        f"lexical_candidates_not_linked={lexical_count:,}"
    )
    return metrics


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def audit(
    args: argparse.Namespace,
    connection: sqlite3.Connection,
    contract_sha256: str,
    export_counts: Mapping[str, int],
    connectivity: Mapping[str, Any],
    progress: TwoLineProgress,
) -> dict[str, Any]:
    progress.start(5, 12, unit="check", estimate=30.0)
    target = args.limit_frames or EXPECTED["frames"]
    counts = {
        "documents": scalar(connection, "SELECT COUNT(*) FROM document"),
        "propositions": scalar(connection, "SELECT COUNT(*) FROM proposition"),
        "assertions": scalar(connection, "SELECT COUNT(*) FROM assertion"),
        "mentions": scalar(connection, "SELECT COUNT(*) FROM mention"),
        "entities": scalar(connection, "SELECT COUNT(*) FROM entity"),
        "binary_edges": scalar(connection, "SELECT COUNT(*) FROM binary_relation"),
        "quarantined_assertions": scalar(connection, "SELECT COUNT(*) FROM quarantined_assertion"),
    }
    progress.update(1)
    checks = {
        "frames_accounted": counts["assertions"] == target,
        "binary_plus_quarantine_accounted": counts["binary_edges"] + counts["quarantined_assertions"] == target,
        "mention_count_matches_stage5": (
            counts["mentions"] == EXPECTED["mentions"] if not args.limit_frames else True
        ),
        "proposition_count_matches_stage4": (
            counts["propositions"] == EXPECTED["propositions"] if not args.limit_frames else True
        ),
        "document_count_matches_stage4": (
            counts["documents"] == EXPECTED["documents"] if not args.limit_frames else True
        ),
    }
    progress.update(1)
    diagnostics = {
        "inconsistent_documents_or_propositions": scalar(
            connection, "SELECT (SELECT COUNT(*) FROM document WHERE consistent=0)+(SELECT COUNT(*) FROM proposition WHERE consistent=0)"
        ),
        "inconsistent_entity_identity": scalar(connection, "SELECT COUNT(*) FROM entity WHERE consistent=0"),
        "dangling_assertion_proposition": scalar(
            connection, "SELECT COUNT(*) FROM assertion a LEFT JOIN proposition p ON p.proposition_id=a.proposition_id WHERE p.proposition_id IS NULL"
        ),
        "dangling_mentions": scalar(
            connection, "SELECT COUNT(*) FROM mention m LEFT JOIN assertion a ON a.assertion_node_id=m.assertion_node_id LEFT JOIN entity e ON e.entity_node_id=m.entity_node_id WHERE a.assertion_node_id IS NULL OR e.entity_node_id IS NULL"
        ),
        "dangling_binary_edges": scalar(
            connection, "SELECT COUNT(*) FROM binary_relation b LEFT JOIN assertion a ON a.assertion_node_id=b.assertion_node_id LEFT JOIN entity s ON s.entity_node_id=b.subject_entity_id LEFT JOIN entity o ON o.entity_node_id=b.object_entity_id WHERE a.assertion_node_id IS NULL OR s.entity_node_id IS NULL OR o.entity_node_id IS NULL"
        ),
        "relation_surface_mutations": scalar(
            connection, "SELECT COUNT(*) FROM binary_relation b JOIN assertion a ON a.assertion_node_id=b.assertion_node_id WHERE b.relation_surface<>a.relation_surface"
        ),
        "binary_self_edges": scalar(connection, "SELECT COUNT(*) FROM binary_relation WHERE subject_entity_id=object_entity_id"),
        "hard_noncanonical_bridge_rows": scalar(
            connection, "SELECT COUNT(*) FROM canonical_entity_document d JOIN entity e ON e.entity_node_id=d.entity_node_id WHERE e.canonical_concept_id IS NULL"
        ),
        "contextual_entity_annotations_missing": scalar(
            connection, "SELECT COUNT(*) FROM mention WHERE entity_node_id='' OR normalization_level=''"
        ),
    }
    for _ in range(8):
        progress.update(1)
    checks.update({key: value == 0 for key, value in diagnostics.items()})
    expected_structural = counts["propositions"] + counts["assertions"] + 2 * counts["mentions"]
    checks["structural_edge_count"] = export_counts.get("structural_edges") == expected_structural
    progress.update(1)
    passed = all(checks.values())
    summary = {
        "version": VERSION,
        "contract_sha256": contract_sha256,
        "scope": {**counts, "full_contract": not bool(args.limit_frames)},
        "stage6": {
            "binary_edge_rate": counts["binary_edges"] / max(1, counts["assertions"]),
            "quarantine_rate": counts["quarantined_assertions"] / max(1, counts["assertions"]),
            "quarantine_reasons": dict(
                connection.execute("SELECT reason,COUNT(*) FROM quarantined_assertion GROUP BY reason").fetchall()
            ),
            "export_counts": dict(export_counts),
        },
        "stage7": dict(connectivity),
        "integrity": {"passed": passed, "checks": checks, "diagnostics": diagnostics},
        "interpretation": {
            "semantic_quality": "Inherited from frozen Stages 2-5; Stage 6-7 add no model prediction.",
            "hard_link_policy": "Only Stage-5 canonical entity_node_id creates cross-document identity.",
            "lexical_policy": "Lexical clusters are diagnostics only and are not graph identity.",
            "next_stage": "Stage 8 constructs question-conditioned bounded subgraphs without modifying this corpus KG.",
        },
    }
    atomic_json(args.output_root / "summary.json", summary)
    if not passed:
        raise RuntimeError(f"Stage-6/7 integrity gate failed: {summary['integrity']}")
    atomic_json(
        args.output_root / "COMPLETE.json",
        {"contract_sha256": contract_sha256, "integrity_passed": True, **counts},
    )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.commit()
    progress.update(1)
    progress.complete(f"integrity_passed=true; summary={args.output_root / 'summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    stages = [
        "validate frozen Stage-5 contract",
        "materialize Stage-6 graph into durable SQLite",
        "export Stage-6 node and edge views",
        "build Stage-7 canonical connectivity indexes",
        "audit invariants and write report",
    ]
    progress = TwoLineProgress(stages)
    # Calibrated from a complete 50,000-frame exact-path run on this NVMe host
    # (7.23 s, 251 MiB RSS).  Full-corpus sorting/export is conservatively
    # budgeted above linear extrapolation.
    progress.estimates = [5.0, 90.0, 150.0, 90.0, 30.0]
    progress.plan(
        [
            "1,034,619 frozen frames and 2,249,489 normalized endpoint mentions",
            "no LLM; exact surface predicates; transaction-resumable CPU/SQLite build",
            "provenance graph plus safe one-subject/one-object binary view",
            "canonical identity hard links only; lexical candidates remain soft",
            "counts, referential integrity, no mutation, no unsafe expansion",
        ]
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / ".run.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another Stage-6/7 process holds {lock_path}") from exc
        try:
            progress.start(1, 5, unit="check", estimate=5.0)
            loaded = validate_stage5(args)
            progress.update(1)
            contract = experiment_contract(args, loaded)
            progress.update(1)
            contract_sha256 = ensure_manifest(args.output_root, contract)
            progress.update(1)
            if args.preflight_only:
                progress.update(2)
                progress.complete(
                    f"preflight passed; target={args.limit_frames or EXPECTED['frames']:,}; output={args.output_root}"
                )
                progress.close(force_complete=False)
                print(f"[preflight complete] output={args.output_root}", flush=True)
                return
            complete = args.output_root / "COMPLETE.json"
            if complete.is_file():
                marker = json.loads(complete.read_text(encoding="utf-8"))
                if marker.get("contract_sha256") == contract_sha256 and marker.get("integrity_passed") is True:
                    progress.update(2)
                    progress.complete(f"cached complete output={args.output_root}")
                    progress.close(force_complete=True)
                    print(f"[workflow complete | cached] output={args.output_root}", flush=True)
                    return
            progress.update(2)
            progress.complete(f"contract={contract_sha256}; output={args.output_root}")
            connection = connect_database(args.output_root / "graph.sqlite3", contract_sha256)
            try:
                materialize(args, connection, progress)
                exports = export_stage6(args, connection, contract_sha256, progress)
                connectivity = build_stage7(args, connection, contract_sha256, progress)
                summary = audit(args, connection, contract_sha256, exports, connectivity, progress)
            finally:
                connection.close()
            progress.close(force_complete=True)
            print(
                f"[workflow complete] frames={summary['scope']['assertions']:,} "
                f"binary_edges={summary['scope']['binary_edges']:,} "
                f"components={summary['stage7']['document_components']:,} "
                f"output={args.output_root}",
                flush=True,
            )
        except Exception as exc:
            progress.log(
                f"[workflow FAILED] stage={progress.index}/{len(stages)} "
                f"completed={progress.done}/{progress.total} error={type(exc).__name__}: {exc}; "
                "rerun the identical command to resume committed Stage-6 frames"
            )
            progress.close(force_complete=False)
            raise


if __name__ == "__main__":
    main()

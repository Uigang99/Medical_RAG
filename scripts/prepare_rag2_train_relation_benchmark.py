#!/usr/bin/env python3
"""Freeze a source-balanced TRAIN reference cohort for paired extraction audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1/reference_statements.jsonl"
RAW = ROOT / "datasets/kg/rag2_statement_graph_v2_pilot_1k_v2/raw_statements_v2.jsonl"
OUTPUT = ROOT / "datasets/kg/rag2_train_relation_reference192_v1"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
SEED = "20260922-train-relation-reference192-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open() as file:
        for line in file:
            yield json.loads(line)


def save(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "train_relation_reference192_v1", "split": "train-only",
        "reference": str(REFERENCE.resolve()), "reference_sha256": sha256(REFERENCE),
        "raw": str(RAW.resolve()), "raw_sha256": sha256(RAW),
        "selection": "40 high-confidence usable plus 8 high-confidence no-core-relation per source, deterministic SHA256 rank. This is a diagnostic Codex-annotated reference, not clinician gold.",
        "seed": SEED, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Train reference contract changed; use a new output root")
    save(path, manifest)
    raw = {r["proposition_id"]: r for r in rows(RAW)}
    pools = defaultdict(list)
    for r in rows(REFERENCE):
        if r["curation_confidence"] != "high" or r["status"] not in ("usable", "no_core_relation"):
            continue
        if r["proposition_id"] not in raw:
            raise RuntimeError("Reference/source mismatch")
        pools[r["source"], r["status"]].append(r)
    selected = []
    for source in SOURCES:
        for status, count in (("usable", 40), ("no_core_relation", 8)):
            ranked = sorted(pools[source, status], key=lambda r: hashlib.sha256((SEED+r["proposition_id"]).encode()).hexdigest())
            if len(ranked) < count:
                raise RuntimeError(f"Insufficient reference cohort {source}/{status}: {len(ranked)}")
            selected.extend(ranked[:count])
    if len(selected) != 192 or len({r["proposition_id"] for r in selected}) != 192:
        raise RuntimeError("Wrong reference cohort")
    cohort = OUTPUT / "cohort.jsonl"
    tmp = cohort.with_suffix(".jsonl.partial")
    with tmp.open("w") as file:
        for r in selected:
            source = raw[r["proposition_id"]]
            item = {"proposition_id": r["proposition_id"], "source": r["source"], "origin": "train_reference",
                    "claim": r["claim"], "evidence": source["evidence"], "reference_status": r["status"]}
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(cohort)
    save(OUTPUT / "selection_summary.json", {"rows": len(selected),
         "by_source_status": dict(Counter(f"{r['source']}/{r['status']}" for r in selected)),
         "cohort_sha256": sha256(cohort), "no_test_data": True})
    print(json.dumps({"output": str(OUTPUT), "rows": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

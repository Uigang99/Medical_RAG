#!/usr/bin/env python3
"""Balanced extraction cohort from the conservative lexical KG candidate tier."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "datasets/kg/rag2_gemma_top8_kg_lexical_tier_v1/propositions.jsonl"
SOURCE_MANIFEST = ROOT / "datasets/kg/rag2_gemma_top8_kg_lexical_tier_v1/manifest.json"
OUTPUT = ROOT / "datasets/kg/rag2_gemma_top8_lexical_relation_pilot192_v1"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
SEED = "20260922-kg-lexical-relation-pilot192-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    if not INPUT.is_file() or not SOURCE_MANIFEST.is_file():
        raise RuntimeError("Run filter_rag2_kg_grounded_propositions.py first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "kg_lexical_relation_pilot192_v1", "source": str(INPUT.resolve()),
                "source_sha256": sha256(INPUT), "source_manifest_sha256": sha256(SOURCE_MANIFEST),
                "cohort": "4 corpora x 3 claim-length strata x (12 original + 4 repaired)=192; one claim per document within stratum",
                "seed": SEED, "test_text_without_test_labels": True,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Cohort contract changed; use a versioned output root")
    save(path, manifest)
    pools = defaultdict(dict)
    counts = Counter()
    with INPUT.open() as file:
        for line in file:
            row = json.loads(line)
            counts["source_rows"] += 1
            key = row["source"], row["origin"], row["claim_length"]
            rank = hashlib.sha256((SEED + row["proposition_id"]).encode()).hexdigest()
            current = pools[key].get(row["document_id"])
            if current is None or rank < current[0]:
                pools[key][row["document_id"]] = rank, row
    selected = []
    for source in SOURCES:
        for length in ("short", "medium", "long"):
            for origin, quota in (("original", 12), ("repair", 4)):
                key = source, origin, length
                ranked = sorted(pools[key].values(), key=lambda pair: pair[0])
                if len(ranked) < quota:
                    raise RuntimeError(f"Insufficient high-grounding stratum {key}: {len(ranked)}")
                selected.extend(row for _, row in ranked[:quota])
    if len(selected) != 192 or len({r["proposition_id"] for r in selected}) != 192:
        raise RuntimeError("Wrong selected cohort")
    cohort = OUTPUT / "cohort.jsonl"
    tmp = cohort.with_suffix(".jsonl.partial")
    with tmp.open("w") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(cohort)
    save(OUTPUT / "selection_summary.json", {"rows": len(selected),
         "source_rows": counts["source_rows"],
         "by_source_origin_length": dict(Counter("/".join((r["source"],r["origin"],r["claim_length"])) for r in selected)),
         "cohort_sha256": sha256(cohort), "not_semantic_gold": True})
    print(json.dumps({"output": str(OUTPUT), "rows": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

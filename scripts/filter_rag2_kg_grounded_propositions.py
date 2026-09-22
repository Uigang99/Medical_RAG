#!/usr/bin/env python3
"""Make a high-lexical-grounding tier from frozen, mechanically valid claims.

This is a precision-oriented gate, not a semantic entailment classifier.
Claims requiring cross-sentence coreference may be excluded even when valid.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "datasets/kg/rag2_gemma_top8_relation_pilot192_v1/machine_eligible_propositions.jsonl"
OUTPUT = ROOT / "datasets/kg/rag2_gemma_top8_kg_lexical_tier_v1"
STOP = frozenset("a an and are as at be been by can could did do does for from had has have if in into is it its may might of on or our per than that the their there these this those to was were which while with would".split())
MIN_CONTENT_WORDS = 3
MIN_COVERAGE = 0.8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_words(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token not in STOP and len(token) >= 3}


def coverage(claim: str, evidence: str) -> tuple[float, int]:
    left, right = content_words(claim), content_words(evidence)
    return len(left & right) / len(left) if left else 0.0, len(left)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "kg_lexical_tier_v1", "source": str(INPUT.resolve()),
                "source_sha256": sha256(INPUT), "minimum_content_words": MIN_CONTENT_WORDS,
                "minimum_claim_content_coverage_in_exact_evidence": MIN_COVERAGE,
                "limitation": "Lexical support only. No entailment, clinical correctness, table alignment, or entity/relation correctness claim."}
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Lexical tier contract changed; use a new output root")
    if not path.exists():
        manifest_tmp = OUTPUT / "manifest.json.partial"
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        manifest_tmp.replace(path)
    tmp = OUTPUT / "propositions.jsonl.partial"
    counts = Counter()
    with INPUT.open() as source, tmp.open("w") as target:
        for line in source:
            row = json.loads(line)
            counts["examined"] += 1
            score, n = coverage(row["claim"], row["evidence"])
            if n < MIN_CONTENT_WORDS or score < MIN_COVERAGE:
                counts["not_high_lexical_grounding"] += 1
                continue
            target.write(json.dumps({**row, "claim_content_coverage_in_evidence": round(score, 4)}, ensure_ascii=False) + "\n")
            counts["retained"] += 1
            counts[f"retained_{row['origin']}"] += 1
            counts[f"retained_{row['source']}"] += 1
    tmp.replace(OUTPUT / "propositions.jsonl")
    summary = {"counts": dict(counts), "output_sha256": sha256(OUTPUT / "propositions.jsonl"),
               "quality_limit": "High lexical grounding is only a conservative candidate gate; source-semantic review required before trusted KG edges."}
    summary_tmp = OUTPUT / "summary.json.partial"
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary_tmp.replace(OUTPUT / "summary.json")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

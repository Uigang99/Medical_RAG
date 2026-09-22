#!/usr/bin/env python3
"""Freeze source-grounded Gemma propositions and a balanced relation pilot."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "datasets/kg/rag2_test6545_restored_top8_gemma_faithful_v3_ctx16k_batch32_v1/document_propositions.jsonl"
REPAIR = ROOT / "datasets/kg/rag2_test6545_restored_top8_gemma_faithful_v3_repair_v1/machine_valid_candidates.jsonl"
OUTPUT = ROOT / "datasets/kg/rag2_gemma_top8_relation_pilot192_v1"
SOURCES = ("cpg", "pmc", "pubmed", "textbooks")
SEED = "20260922-gemma-relation-pilot-v1"
QUOTA = {"original": 12, "repair": 4}  # Per source and claim-length stratum.


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def main() -> None:
    if not ORIGINAL.is_file() or not REPAIR.is_file():
        raise RuntimeError("Required original or repair result is missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "gemma_top8_relation_pilot192_v1",
        "hypothesis": "Specific biomedical participants and text-grounded semantic relations can be extracted from standalone propositions; exact evidence alone is not semantic gold.",
        "original": str(ORIGINAL.resolve()), "original_sha256": file_digest(ORIGINAL),
        "repair": str(REPAIR.resolve()), "repair_sha256": file_digest(REPAIR),
        "selection": "Replace only machine-valid repaired documents; require recovered source, clean document-level parse, exact source evidence, and no decontextualization flag. Exclude source-quality warnings. One claim per document in each sampled stratum.",
        "pilot": "Four corpora x three claim lengths (<=20, 21-40, >40 words) x (12 original + 4 repaired) = 192 claims. Deliberately balanced, not representative of corpus prevalence.",
        "seed": SEED, "models": ["Qwen3.5-9B", "gemma-4-31B-it"],
        "no_qa_labels_or_question_input": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    mf = OUTPUT / "manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise RuntimeError("Pilot selection contract changed; use a new output root")
    atomic_json(mf, manifest)

    repair = {}
    for row in load_rows(REPAIR):
        sid = row["stable_id"]
        if sid in repair or not row["machine_valid"]:
            raise RuntimeError("Invalid repair candidate set")
        repair[sid] = row
    if len(repair) != 946:
        raise RuntimeError(f"Unexpected repair count: {len(repair)}")

    counters = Counter()
    pools: dict[tuple[str, str, str], dict[str, tuple[str, dict]]] = defaultdict(dict)
    seen_repair = set()
    tmp = (OUTPUT / "machine_eligible_propositions.jsonl.partial")
    with tmp.open("w") as target:
        for original in load_rows(ORIGINAL):
            counters["source_documents"] += 1
            sid = original["stable_id"]
            if original["source_status"] != "boundary_expanded_candidate" or original["source_quality_flags"]:
                counters["excluded_source_warning_documents"] += 1
                continue
            replacement = repair.get(sid)
            if replacement is not None:
                seen_repair.add(sid)
                if replacement["text_sha256"] != original["text_sha256"]:
                    raise RuntimeError(f"Repair/source hash mismatch: {sid}")
                propositions = replacement["parsed"]["propositions"]
                origin = "repair"
            else:
                if original["parse_status"] != "parsed" or original["parse_errors"] or original["finish_reason"] == "length":
                    counters["excluded_unresolved_documents"] += 1
                    continue
                propositions = original["propositions"]
                origin = "original"
            counters[f"selected_{origin}_documents"] += 1
            for index, prop in enumerate(propositions):
                if not prop["evidence_exact_match"] or prop.get("decontextualization_flag"):
                    counters["excluded_proposition_warning"] += 1
                    continue
                claim, evidence = prop["claim"].strip(), prop["evidence"].strip()
                if not claim or not evidence or evidence not in original["source_text"]:
                    raise RuntimeError(f"Invalid supposedly exact proposition: {sid}/{index}")
                claim_words = len(claim.split())
                length = "short" if claim_words <= 20 else "medium" if claim_words <= 40 else "long"
                item = {"proposition_id": digest(f"{sid}|{origin}|{index}|{claim}|{evidence}"),
                        "document_id": sid, "source": original["source"], "origin": origin,
                        "claim_length": length, "claim": claim, "evidence": evidence,
                        "source_text_sha256": original["text_sha256"]}
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
                counters[f"eligible_{origin}_propositions"] += 1
                bucket = (item["source"], origin, length)
                rank = digest(SEED + "|" + item["proposition_id"])
                current = pools[bucket].get(sid)
                if current is None or rank < current[0]:
                    pools[bucket][sid] = (rank, item)
    if counters["source_documents"] != 44911:
        raise RuntimeError("Wrong original Top-8 document count")
    if not seen_repair.issubset(repair):
        raise RuntimeError("Unexpected repair IDs")
    tmp.replace(OUTPUT / "machine_eligible_propositions.jsonl")

    selected = []
    for source in SOURCES:
        for origin, count in QUOTA.items():
            for length in ("short", "medium", "long"):
                key = (source, origin, length)
                ranked = sorted(pools[key].values(), key=lambda pair: pair[0])
                if len(ranked) < count:
                    raise RuntimeError(f"Insufficient pilot stratum {key}: {len(ranked)} < {count}")
                for _, item in ranked[:count]:
                    selected.append(item)
    if len(selected) != 192 or len({r["proposition_id"] for r in selected}) != 192:
        raise RuntimeError("Pilot size/identity failure")
    cohort = OUTPUT / "cohort.jsonl"
    tmp = cohort.with_suffix(".jsonl.partial")
    with tmp.open("w") as out:
        for item in selected:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(cohort)
    summary = {"counts": dict(counters), "pilot_rows": len(selected),
               "pilot_strata": {"/".join(k): len(v) for k, v in pools.items()},
               "machine_eligible_not_semantic_gold": True,
               "cohort_sha256": file_digest(cohort)}
    atomic_json(OUTPUT / "selection_summary.json", summary)
    print(json.dumps({"output": str(OUTPUT), "eligible": counters["eligible_original_propositions"] + counters["eligible_repair_propositions"],
                      "pilot": len(selected), "repaired_documents_used": counters["selected_repair_documents"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a non-destructive PMID/title eligibility view from the audited corpus.

The existing audit stores a PMID, text hash, and title-only bit per original row.
It is reused only after shard alignment, timestamp and sampled metadata checks.
All decisions are question-independent; no benchmark answers are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import xxhash

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from medrag.retrieval.corpus_filter import select_rows, sha256_file, signature
from medrag.training.graph_input_progress import Progress
from scripts.build_question_only_fourway_retrieval import atomic_json, checked_manifest

SOURCES = ("pubmed", "pmc", "cpg", "textbooks")
EXPECTED = {"pubmed": 2346290, "pmc": 45736005, "cpg": 721353, "textbooks": 133875}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--vector-db-root", type=Path, default=PROJECT / "databases/vector_db/RAG_Square")
    p.add_argument("--output-root", type=Path,
                   default=PROJECT / "databases/corpus_filters/pmc_pmid_no_pubmed_title_v1")
    return p.parse_args()


def verify_audit(path, metadata, offsets_path, rows, start, seed):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.shape != (rows,) or data.dtype.names != ("hash", "pmid", "flags"):
        raise RuntimeError(f"Wrong audit array: {path}")
    if metadata.stat().st_mtime_ns > path.stat().st_mtime_ns:
        raise RuntimeError(f"Metadata newer than audit: {metadata}")
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if offsets.shape != (rows,):
        raise RuntimeError(f"Wrong metadata offsets: {offsets_path}")
    rng = np.random.default_rng(seed)
    positions = np.unique(np.r_[0, rows - 1, rng.integers(0, rows, size=30)])
    with metadata.open("rb") as handle:
        for i in positions:
            handle.seek(int(offsets[i]))
            row = json.loads(handle.readline())
            raw_pmid = (row.get("metadata") or {}).get("pmid")
            pmid = int(raw_pmid) if raw_pmid and str(raw_pmid).isdigit() else 0
            text = " ".join((row.get("text") or "").split())
            title = " ".join((row.get("title") or "").split())
            if (row["local_id"] != start + i or int(data[i]["pmid"]) != pmid or
                bytes(data[i]["hash"]) != xxhash.xxh3_128_digest(text.encode()) or
                bool(data[i]["flags"] & 1) != bool(title and title == text)):
                raise RuntimeError(f"Audit/metadata mismatch: {metadata}:{i}")
    return data


def run(a):
    progress = Progress(["validate audit/source", "collect PMC PMIDs", "write eligibility masks", "verify counts"])
    arrays, sources, inputs = {}, {}, []
    print("[plan] verify immutable audit -> PMC PMID set -> source-local eligibility masks -> counts; "
          "CPU only; no corpus/vector deletion; no question or gold labels", flush=True)
    try:
        manifests = {s: json.loads((a.vector_db_root / s / "manifest.json").read_text()) for s in SOURCES}
        count = sum(len(m["metadata"]["shards"]) for m in manifests.values())
        progress.stage(0, count, unit="shard")
        for source, manifest in manifests.items():
            root = a.vector_db_root / source
            manifest_path = root / "manifest.json"
            metadata_paths, offset_paths = [], []
            arrays[source] = []
            cursor = 0
            for number, shard in enumerate(manifest["metadata"]["shards"]):
                path = a.audit_root / f"{source}_{number:03d}.npy"
                stats = json.loads(path.with_suffix(".json").read_text())
                rows, start = int(shard["rows"]), int(shard["start_local_id"])
                if (start != cursor or stats["rows"] != rows or stats["start"] != start or
                    stats["source"] != source or stats["shard"] != number):
                    raise RuntimeError(f"Audit shard alignment mismatch: {path}")
                metadata = root / shard["path"]
                offsets = metadata.with_suffix(".offsets.npy")
                data = verify_audit(path, metadata, offsets, rows, start, seed=20261029 + number)
                arrays[source].append((start, data))
                metadata_paths.append(metadata)
                offset_paths.append(offsets)
                inputs.append({**signature(path), "sha256": sha256_file(path)})
                cursor += rows
                progress.update(1)
            if cursor != manifest["rows"]:
                raise RuntimeError(f"Wrong source row total: {source}")
            sources[source] = {"rows": cursor, "manifest_sha256": sha256_file(manifest_path),
                               "shards": [signature(root / s["path"]) for s in manifest["index"]["shards"]],
                               "metadata": [signature(p) for p in metadata_paths],
                               "metadata_offsets": [signature(p) for p in offset_paths]}
        contract = {"version": 1, "sources": sources, "audited_inputs": inputs,
                    "rules": {"allowed_articles": "all nonzero PMIDs present in current PMC",
                              "pubmed": "PMID in allowed_articles AND NOT normalized(text)==normalized(nonempty title)",
                              "pmc": "PMID in allowed_articles; missing PMID excluded",
                              "cpg_textbooks": "unchanged; no title filtering",
                              "normalization": "collapse whitespace only",
                              "preserve": "original source/local_id/chunk_id/PMID/text/embedding"},
                    "validation": "full audit arrays; per-shard row/time checks and 32 deterministic metadata samples",
                    "code_sha256": {str(p.relative_to(PROJECT)): sha256_file(p) for p in
                                    (Path(__file__), PROJECT / "medrag/retrieval/corpus_filter.py")}}
        contract["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        checked_manifest(a.output_root, contract)
        progress.complete(f"verified {count} audit shards; {a.output_root / 'manifest.json'}")

        progress.stage(1, len(arrays["pmc"]), unit="shard")
        parts = []
        for _, data in arrays["pmc"]:
            pmids = np.unique(data["pmid"])
            parts.append(pmids[pmids != 0])
            progress.update(1)
        allowed = np.unique(np.concatenate(parts))
        if len(allowed) != 1007780:
            raise RuntimeError(f"PMC distinct PMID count changed: {len(allowed)}")
        pmid_path = a.output_root / "allowed_pmids.npy"
        with pmid_path.with_suffix(".npy.tmp").open("wb") as f:
            np.save(f, allowed)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pmid_path.with_suffix(".npy.tmp"), pmid_path)
        progress.complete(f"{len(allowed):,} valid PMIDs; {pmid_path}")

        progress.stage(2, count, unit="shard")
        result = {}
        for source in SOURCES:
            path = a.output_root / f"{source}.npy"
            tmp = path.with_suffix(".npy.tmp")
            mask = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.bool_, shape=(sources[source]["rows"],))
            for start, data in arrays[source]:
                mask[start:start + len(data)] = select_rows(source, data["pmid"], data["flags"], allowed)
                progress.update(1)
            eligible = int(np.count_nonzero(mask))
            if eligible != EXPECTED[source]:
                raise RuntimeError(f"Eligibility count mismatch: {source}: {eligible} != {EXPECTED[source]}")
            mask.flush()
            del mask
            with tmp.open("rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, path)
            result[source] = {"rows": sources[source]["rows"], "eligible_rows": eligible,
                              "excluded_rows": sources[source]["rows"] - eligible,
                              "mask": path.name, "sha256": sha256_file(path)}
        progress.complete(f"four immutable-source masks; {a.output_root}")

        progress.stage(3, 1, unit="check")
        summary = {"contract_sha256": contract["contract_sha256"], "allowed_pmids": len(allowed),
                   "sources": result, "eligible_rows": sum(v["eligible_rows"] for v in result.values()),
                   "original_rows": sum(v["rows"] for v in result.values()), "source_data_changed": False}
        atomic_json(a.output_root / "summary.json", summary)
        progress.complete(f"verified {summary['eligible_rows']:,} eligible; {a.output_root / 'summary.json'}")
        progress.log(f"[complete] {summary['eligible_rows']:,} eligible chunks; all original artifacts preserved")
    except Exception as e:
        progress.log(f"[FAILED] stage={progress.index + 1}/4 completed={progress.done}/{progress.total}; "
                     f"output={a.output_root}; original corpus unchanged; rerun revalidates/rebuilds masks; {e}")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    run(arguments())

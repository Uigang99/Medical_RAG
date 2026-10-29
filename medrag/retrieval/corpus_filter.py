"""Immutable source-local eligibility masks; no text/vector rewriting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def select_rows(source: str, pmids: np.ndarray, flags: np.ndarray,
                allowed_pmids: np.ndarray) -> np.ndarray:
    if source in ("cpg", "textbooks"):
        return np.ones(len(pmids), dtype=np.bool_)
    if source not in ("pubmed", "pmc"):
        raise ValueError(source)
    keep = (pmids != 0) & np.isin(pmids, allowed_pmids, kind="table")
    if source == "pubmed":
        keep &= (flags & 1) == 0
    return keep


def load_corpus_filter(root: Path, source_contract: list[dict]) -> tuple[dict, dict]:
    """Fail before loading vectors if masks or the underlying corpus changed."""
    manifest_path, summary_path = root / "manifest.json", root / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary.get("contract_sha256") != manifest["contract_sha256"]:
        raise RuntimeError("Incomplete corpus eligibility contract")
    masks = {}
    for source in source_contract:
        name = source["source"]
        frozen = manifest["sources"][name]
        for key in ("rows", "manifest_sha256", "shards", "metadata", "metadata_offsets"):
            if frozen[key] != source[key]:
                raise RuntimeError(f"Corpus changed after eligibility construction: {name}/{key}")
        item = summary["sources"][name]
        path = root / item["mask"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Eligibility checksum mismatch: {path}")
        mask = np.load(path, mmap_mode="r", allow_pickle=False)
        if (mask.dtype != np.bool_ or mask.shape != (source["rows"],) or
                int(np.count_nonzero(mask)) != item["eligible_rows"]):
            raise RuntimeError(f"Invalid eligibility mask: {path}")
        masks[name] = mask
    return masks, {"path": str(root.resolve()), "manifest_sha256": sha256_file(manifest_path),
                   "summary_sha256": sha256_file(summary_path),
                   "rules": manifest["rules"], "eligible_rows": summary["eligible_rows"],
                   "by_source": {s: v["eligible_rows"] for s, v in summary["sources"].items()}}

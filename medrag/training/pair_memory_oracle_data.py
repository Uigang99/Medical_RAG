"""Immutable official-train cohorts and exact, untruncated pair encoder inputs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from medrag.generation.pair_memory_attention import pair_encoding
from medrag.training.graph_input_data import digest, identity, jsonl, sha_file, normalize_question
from medrag.training.proposition_kg_attention_data import shuffle_pairs


def stable(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def variants(row, seed):
    return {"kg": row, "rewired": shuffle_pairs(row, seed)}


def encode_specs(tokenizer, row, seed):
    result = {}
    for mode, layout in variants(row, seed).items():
        records = []
        for a, b in layout["pairs"]:
            encoded, ai, bi = pair_encoding(tokenizer, row["question"],
                                            row["claims"][a], row["claims"][b])
            records.append({"pair": [a, b], "encoding": encoded, "a_tokens": ai, "b_tokens": bi})
        result[mode] = records
    return result


def prepare(a, progress=None):
    """CPU only. Select by dataset proportion/hash, never correctness."""
    donor_path = a.prepared_root / "manifest.json"
    donor = json.loads(donor_path.read_text())
    previous = json.loads((a.source_root / "manifest.json").read_text())
    complete = json.loads((a.prepared_root / "prepared/COMPLETE.json").read_text())
    for obj in (donor, previous):
        if digest(obj["contract"]) != obj["contract_sha256"]:
            raise ValueError("Invalid input manifest hash")
    if (previous["contract"]["source_manifest"] != donor
            or complete["contract_sha256"] != donor["contract_sha256"]
            or donor["contract"].get("lora") is not None):
        raise ValueError("Require original pristine-Llama official-train artifacts")
    row_path = a.prepared_root / "prepared/rows.jsonl"
    rh = sha_file(row_path)
    if rh != complete["rows_sha256"] or rh != previous["contract"]["prepared_sha256"]:
        raise ValueError("Prepared source rows changed")
    for meta in donor["contract"]["inputs"]["model_files"].values():
        if identity(meta["path"], full_hash="sha256" in meta) != meta:
            raise ValueError("Base model/tokenizer changed: " + meta["path"])
    prior_rows = list(jsonl(a.source_root / "capability_interventions.jsonl"))
    prior = {r["sample_id"]: r for r in prior_rows}
    if (len(prior_rows) != 128 or len(prior) != 128
            or sum(bool(r["base"]["correct"]) for r in prior_rows) != 96):
        raise ValueError("Expected unchanged exploratory TRAIN128 (96 correct/32 wrong)")
    if any(r["contract_sha256"] != previous["contract_sha256"] for r in prior_rows):
        raise ValueError("Diagnostic records have wrong contract")
    rows = list(jsonl(row_path))
    if Counter(r["split"] for r in rows) != {"train": 9000, "validation": 1000}:
        raise ValueError("Not the official train9000/validation1000 dataset")
    source = Path(donor["contract"]["inputs"]["source_root"])
    views = {r["sample_id"]: r for r in jsonl(source / "question_views.jsonl")}
    diagnosis = []
    for row in rows:
        view = views[row["sample_id"]]
        if view["official_split"] != "train" or view["split"] != row["split"]:
            raise ValueError("Benchmark TEST or split mismatch")
        row["question"] = view["question"]
        if row["sample_id"] in prior:
            original = {k: v for k, v in row.items() if k != "question"}
            if digest(dict(original, diagnostic_split="capability")) != prior[row["sample_id"]]["row_sha256"]:
                raise ValueError("Fixed diagnostic prompt/layout changed")
            diagnosis.append(row)
    if len(diagnosis) != 128:
        raise ValueError("Missing diagnostic rows")
    diagnosis.sort(key=lambda r: stable(a.seed, r["sample_id"]))
    blocked = {normalize_question(r["question"]) for r in diagnosis}
    # Bounded candidate pools; long pairs are excluded BEFORE answer inspection.
    pools = {}
    for split, count in (("train", a.train_questions), ("validation", a.dev_questions)):
        pools[split] = {}
        medmcqa_count = round(count * .7)
        for dataset, need in (("medmcqa", medmcqa_count), ("medqa", count - medmcqa_count)):
            candidates = [r for r in rows if r["split"] == split and r["dataset"] == dataset
                          and normalize_question(r["question"]) not in blocked and r["pairs"]]
            candidates.sort(key=lambda r: stable(a.seed, r["sample_id"]))
            pools[split][dataset] = (need, candidates[:max(2 * need, need + 32)])
    selected_pool = diagnosis + [r for pool in pools.values() for _, rs in pool.values() for r in rs]
    wanted = {pid for r in selected_pool for pid in r["proposition_ids"]}
    claims = {}
    for p in jsonl(source / "usable/usable_propositions.jsonl"):
        if p["proposition_id"] in wanted:
            if p["proposition_id"] in claims or not p["claim"].strip():
                raise ValueError("Duplicate/empty source claim")
            claims[p["proposition_id"]] = p["claim"]
    if set(claims) != wanted:
        raise ValueError("Missing exact source propositions")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.encoder, local_files_only=True, use_fast=True)
    llama_tokenizer = AutoTokenizer.from_pretrained(donor["contract"]["model"], local_files_only=True,
                                                    clean_up_tokenization_spaces=False)
    specs, rejected = {}, []

    def verify(row):
        row["claims"] = [claims[pid] for pid in row["proposition_ids"]]
        # These cached spans can include a leading space but not different facts.
        for claim, (s, e) in zip(row["claims"], row["proposition_spans"]):
            decoded = llama_tokenizer.decode(row["input_ids"][s:e], skip_special_tokens=False)
            if " ".join(decoded.split()) != " ".join(claim.split()):
                raise ValueError(f"Source claim/token span differs: {row['sample_id']}")
        return encode_specs(tokenizer, row, a.seed)

    for row in diagnosis:
        specs[row["sample_id"]] = verify(row)  # Never silently drop diagnosis rows.
        if progress:
            progress.update()
    groups = {"diagnosis": diagnosis}
    for split, output_name in (("train", "train"), ("validation", "dev")):
        chosen = []
        for dataset, (need, candidates) in pools[split].items():
            found = 0
            for row in candidates:
                stem = normalize_question(row["question"])
                if stem in blocked:
                    continue
                try:
                    value = verify(row)
                except ValueError as exc:
                    if not str(exc).startswith("MedCPT pair has"):
                        raise
                    rejected.append({"sample_id": row["sample_id"], "reason": str(exc)})
                    continue
                specs[row["sample_id"]] = value
                chosen.append(row)
                blocked.add(stem)
                found += 1
                if progress:
                    progress.update()
                if found == need:
                    break
            if found != need:
                raise ValueError(f"Not enough untruncated {split}/{dataset}: {found}/{need}")
        groups[output_name] = sorted(chosen, key=lambda r: stable(a.seed, r["sample_id"]))
    used = [r for group in groups.values() for r in group]
    if len({r["sample_id"] for r in used}) != len(used):
        raise ValueError("Cohort ID overlap")
    if set(normalize_question(r["question"]) for r in groups["train"]) & set(
            normalize_question(r["question"]) for r in groups["dev"]):
        raise ValueError("Train/dev exact-normalized stem overlap")
    metadata = {
        "source_contract": donor["contract_sha256"], "prepared_sha256": rh,
        "diagnosis_source": identity(a.source_root / "capability_interventions.jsonl", full_hash=True),
        "exact_text_sources": [identity(source / p, full_hash=True)
                               for p in ("question_views.jsonl", "usable/usable_propositions.jsonl")],
        "model": donor["contract"]["model"], "model_files": donor["contract"]["inputs"]["model_files"],
        "cohorts": {name: [{"id": r["sample_id"], "dataset": r["dataset"], "row_hash": digest(r)}
                           for r in group] for name, group in groups.items()},
        "selection": "dataset-proportional seeded hash; graph-bearing bridge train/dev; no correctness selection",
        "excluded_long_pairs": rejected, "no_truncation": True,
        "diagnosis_caveat": "Previously analyzed official TRAIN128; exploratory, not fresh held-out or benchmark TEST",
        "duplicate_check": "Exact normalized stems excluded across cohorts; semantic duplicates not certified",
        "encoder_inputs_hash": digest(specs),
        "counts": {name: dict(Counter(r["dataset"] for r in group)) for name, group in groups.items()},
        "pair_counts": {name: sum(len(r["pairs"]) for r in group) for name, group in groups.items()},
    }
    return groups, specs, metadata, prior

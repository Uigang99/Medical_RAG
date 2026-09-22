#!/usr/bin/env python3
"""Compare paired predictions with the existing TRAIN-only source-reviewed reference.

Endpoint-only overlap is a diagnostic, not semantic relation precision: the
reference relation ontology omits valid free-text relations and concept linking.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "datasets/kg/rag2_statement_graph_reference_1k_codex_v1/reference_statements.jsonl"
RESULT = ROOT / "results/rag2_train_relation_reference192_v1"
MODELS = ("Qwen3.5-9B", "gemma-4-31B-it")


def rows(path: Path):
    with path.open() as file:
        for line in file:
            yield json.loads(line)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def match(pred: str, gold: str, relaxed: bool = False) -> bool:
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    if not relaxed or not a or not b:
        return False
    aa, bb = set(a.split()), set(b.split())
    return len(aa & bb) / max(len(aa | bb), 1) >= 0.75


def score(row: dict, gold: dict, model: str, relaxed: bool) -> dict:
    extracted = row["models"][model]
    preds = extracted["entities"]
    gold_entities = gold["entities"]
    taken = set()
    mapping = {}
    for pred in preds:
        choices = [g for g in gold_entities if g["entity_id"] not in taken and match(pred["span"], g["surface"], relaxed)]
        if not choices:
            continue
        chosen = max(choices, key=lambda g: len(normalize(g["surface"])))
        mapping[pred["id"]] = chosen["entity_id"]
        taken.add(chosen["entity_id"])
    matched = {(mapping.get(r["subject_id"]), mapping.get(r["object_id"]))
               for r in extracted["relations"]}
    gold_pairs = {(r["subject_entity_id"], r["object_entity_id"]) for r in gold["relations"]}
    return {"pred_entities": len(preds), "gold_entities": len(gold_entities),
            "matched_entities": len(mapping), "pred_relations": len(extracted["relations"]),
            "gold_relations": len(gold["relations"]), "matched_oriented_endpoint_pairs": len(matched & gold_pairs),
            "positive_relation_output": bool(extracted["relations"])}


def summarize(data: list[dict]) -> dict:
    count = Counter()
    for r in data:
        for key in ("pred_entities", "gold_entities", "matched_entities", "pred_relations", "gold_relations", "matched_oriented_endpoint_pairs"):
            count[key] += r[key]
        count["rows"] += 1
        count["nonempty_relation_rows"] += r["positive_relation_output"]
    result = dict(count)
    result["entity_exact_precision"] = count["matched_entities"] / count["pred_entities"] if count["pred_entities"] else None
    result["entity_exact_recall"] = count["matched_entities"] / count["gold_entities"] if count["gold_entities"] else None
    result["oriented_pair_overlap_per_gold_relation"] = count["matched_oriented_endpoint_pairs"] / count["gold_relations"] if count["gold_relations"] else None
    return result


def main() -> None:
    refs = {r["proposition_id"]: r for r in rows(REFERENCE)}
    paired = list(rows(RESULT / "paired_review.jsonl"))
    if len(paired) != 192 or len({r["proposition_id"] for r in paired}) != 192:
        raise RuntimeError("Incomplete or repeated TRAIN paired results")
    full = defaultdict(list)
    per_row = []
    for row in paired:
        gold = refs[row["proposition_id"]]
        item = {"proposition_id": row["proposition_id"], "source": row["source"], "reference_status": gold["status"], "models": {}}
        for model in MODELS:
            strict = score(row, gold, model, False)
            relaxed = score(row, gold, model, True)
            item["models"][model] = {"strict": strict, "relaxed": relaxed}
            for variant, metrics in (("strict", strict), ("relaxed", relaxed)):
                full[model, variant, "all"].append(metrics)
                full[model, variant, row["source"]].append(metrics)
                full[model, variant, gold["status"]].append(metrics)
        per_row.append(item)
    summary = {model: {variant: {scope: summarize(items) for (m, v, scope), items in full.items() if m == model and v == variant}
                       for variant in ("strict", "relaxed")} for model in MODELS}
    rng = random.Random(20260922)
    values = []
    for _ in range(10000):
        sampled = [per_row[rng.randrange(len(per_row))] for _ in per_row]
        denominator = sum(r["models"][MODELS[0]]["strict"]["gold_relations"] for r in sampled)
        difference = sum(r["models"][MODELS[1]]["strict"]["matched_oriented_endpoint_pairs"] -
                         r["models"][MODELS[0]]["strict"]["matched_oriented_endpoint_pairs"] for r in sampled)
        values.append(difference / denominator if denominator else 0.0)
    values.sort()
    summary["paired_bootstrap_gemma_minus_qwen_oriented_pair_recall"] = {
        "estimate": (summary[MODELS[1]]["strict"]["all"]["matched_oriented_endpoint_pairs"] -
                     summary[MODELS[0]]["strict"]["all"]["matched_oriented_endpoint_pairs"]) /
                    summary[MODELS[0]]["strict"]["all"]["gold_relations"],
        "95pct_interval": [values[250], values[9749]], "unit": "absolute fraction", "resampling_unit": "proposition"}
    summary["interpretation"] = "Exact/relaxed entity span agreement and raw ordered endpoint-pair overlap against Codex-curated TRAIN reference. Because raw predicates can be inverse phrases (e.g. X is a consequence of Y vs Y CAUSES X), pair overlap is NOT semantic direction accuracy, full relation precision/recall, clinician gold, or test performance."
    for filename, value in (("reference_benchmark.json", summary),):
        tmp = (RESULT / filename).with_suffix(".json.partial")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        tmp.replace(RESULT / filename)
    tmp = RESULT / "reference_benchmark_per_row.jsonl.partial"
    with tmp.open("w") as file:
        for row in per_row:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(RESULT / "reference_benchmark_per_row.jsonl")
    print(json.dumps({model: {v: summary[model][v]["all"] for v in ("strict", "relaxed")} for model in MODELS}, ensure_ascii=False))


if __name__ == "__main__":
    main()

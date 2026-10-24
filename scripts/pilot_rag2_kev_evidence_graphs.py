#!/usr/bin/env python3
"""Bounded, exploratory TEST pilot: fixed propositions vs KEV graph annotations.

No training, no new retrieval, no generated intermediate conclusions. GPU
workers are sequential processes so KEV's memory is released before vLLM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import evidence_relation_graph as g
from medrag.benchmark import load_benchmark_samples, resolve_benchmark_path
from medrag.evidence_graph_runtime import DecisionCache, KEV_MAX_ROW, KEV_MAX_STATE
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, jsonl
from medrag.training.graph_input_progress import Progress

EVENT = "@@EVIDENCE_GRAPH@@"
STAGES = ("contract / CPU prepare", "load KEV", "A+C labels", "B claim labels",
          "question expert", "group expert", "load Llama", "paired generation", "paired report")
DEFAULT_SOURCE = ROOT / "results/rag2_route_feedback_kev_test6545_v2/cohort.jsonl"
DEFAULT_GRAPH = ROOT / "datasets/kg/rag2_test6545_union88294_stage7_surface_kg_v1/graph.sqlite3"


class PilotProgress(Progress):
    """Persist real stage durations; reuse only this exact run's compatible rates."""
    def bind(self, root, contract_hash):
        self.runtime_path = root / "RUNTIME.json"
        self.contract_hash = contract_hash
        self.history = []
        if self.runtime_path.exists():
            previous = json.loads(self.runtime_path.read_text())
            if previous["contract_hash"] != contract_hash:
                raise RuntimeError("Runtime timing contract mismatch")
            self.history = previous["stages"]
            for r in self.history:
                if r["fresh_units"] > 0:
                    self.weights[r["index"]] = r["elapsed_seconds"] * r["total_units"] / r["fresh_units"]

    def complete(self, message):
        super().complete(message)
        if hasattr(self, "runtime_path"):
            self.history.append({"index": self.index, "name": self.names[self.index],
                                 "elapsed_seconds": time.monotonic()-self.stage_started,
                                 "fresh_units": self.total-self.initial, "cached_units": self.initial,
                                 "total_units": self.total, "unit": self.unit, "finished_unix": time.time()})
            atomic_json(self.runtime_path, {"contract_hash": self.contract_hash, "stages": self.history})


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true", help="CPU checks only; no model/GPU loading or output writes")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--report", action="store_true")
    modes.add_argument("--worker", choices=("kev", "generate"), help=argparse.SUPPRESS)
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_kev_evidence_graphs_test64_v1")
    p.add_argument("--source-cohort", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    p.add_argument("--benchmark-root", type=Path, default=ROOT / "datasets/benchmark")
    p.add_argument("--ontology", type=Path, help="Optional verified ontology JSON; absent means ABOUT graph without hierarchy")
    p.add_argument("--questions", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--kev-source", type=Path, default=ROOT.parent / "tools/kev-runtime")
    p.add_argument("--kev-adapter", type=Path, default=ROOT.parent / "models/kev-9b")
    p.add_argument("--kev-base", type=Path, default=ROOT.parent / "models/Qwen3.5-9B-Base")
    p.add_argument("--entities-per-proposition", type=int, default=2)
    p.add_argument("--pair-neighbors", type=int, default=2)
    p.add_argument("--max-topics", type=int, default=16)
    p.add_argument("--max-groups", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--confidence", type=float, default=.7)
    p.add_argument("--margin", type=float, default=.15)
    p.add_argument("--kev-batch-rows", type=int, default=32)
    p.add_argument("--generation-batch", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=640)
    p.add_argument("--annotation-tokens", type=int, default=1536)
    p.add_argument("--gpu-memory-utilization", type=float, default=.85)
    a = p.parse_args(argv)
    if not (8 <= a.questions <= 256 and 1 <= a.entities_per_proposition <= 4 and 1 <= a.pair_neighbors <= 4
            and 2 <= a.max_topics <= 32 and 1 <= a.max_groups <= 4 and 2 <= a.group_size <= 4
            and .5 <= a.confidence <= 1 and 0 <= a.margin <= 1 and 8 <= a.kev_batch_rows <= 128
            and 1 <= a.generation_batch <= 64 and 4096 <= a.max_model_len <= 8192
            and 128 <= a.max_new_tokens <= 1024 and 256 <= a.annotation_tokens <= 2048
            and .3 <= a.gpu_memory_utilization <= .9):
        p.error("Invalid pilot/resource bounds")
    return a


def settings(a):
    return {k: getattr(a, k) for k in ("entities_per_proposition", "pair_neighbors", "max_topics",
            "max_groups", "group_size", "confidence", "margin")}


def emit(event, **data):
    print(EVENT + json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def item_path(root, kind, sample_id):
    key = digest(sample_id)
    return root / kind / key[:2] / (key + ".json")


def selected_cohort(a):
    metadata = [{"sample_id": r["sample_id"], "dataset": r["dataset"]} for r in jsonl(a.source_cohort)]
    if len(metadata) != 6545:
        raise RuntimeError("Expected original, unpruned TEST6545 cohort")
    chosen = g.select_sample(metadata, a.questions, a.seed)
    wanted = {r["sample_id"] for r in chosen}
    selected = {r["sample_id"]: r for r in jsonl(a.source_cohort) if r["sample_id"] in wanted}
    benchmark, sources = {}, {}
    for dataset in sorted({r["dataset"] for r in chosen}):
        path = resolve_benchmark_path(a.benchmark_root, "mcq", "unified", dataset, "test")
        sources[dataset] = identity(path, full_hash=True)
        benchmark.update({s.id: s for s in load_benchmark_samples(path, "mcq", "unified", dataset, "test") if s.id in wanted})
    rows, labels = [], {}
    for meta in chosen:
        r, s = selected[meta["sample_id"]], benchmark[meta["sample_id"]]
        opts = {k.upper(): v for k, v in (s.options or {}).items()}
        formatted = "\n".join([s.question] + [f"{k}. {v}" for k, v in sorted(opts.items())])
        if r["question"] != formatted or s.dataset != r["dataset"] or s.split != "test":
            raise RuntimeError(f"Source/benchmark question mismatch: {s.id}")
        if len(r["documents"]) != 8 or [d["rank"] for d in r["documents"]] != list(range(1, 9)):
            raise RuntimeError(f"Top-8 contract failed: {s.id}")
        if len({d["document_id"] for d in r["documents"]}) != 8:
            raise RuntimeError(f"Duplicate retrieved document: {s.id}")
        ps = g.propositions(r)
        if len({p["proposition_id"] for p in ps}) != len(ps) or any(not p["claim"].strip() for p in ps):
            raise RuntimeError(f"Invalid proposition identities/text: {s.id}")
        if opts and s.answer not in opts:
            raise RuntimeError(f"Unexpected evaluation gold format: {s.id}")
        # Gold stays in a separate artifact; never passed to model workers.
        labels[s.id] = s.answer
        rows.append({"sample_id": s.id, "dataset": s.dataset, "question": s.question,
                     "options": opts, "documents": r["documents"]})
    return rows, labels, sources


def build_index(rows, graph_path):
    ps = {}
    for row in rows:
        for doc in row["documents"]:
            for p in doc["propositions"]:
                value = {"claim": p["claim"], "document_id": doc["document_id"], "entities": {}}
                if p["proposition_id"] in ps and ps[p["proposition_id"]] != value:
                    raise RuntimeError("Same proposition ID has different text/source")
                ps[p["proposition_id"]] = value
    entities, triples, missing = {}, [], []
    with sqlite3.connect(f"file:{graph_path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        ids = sorted(ps)
        for start in range(0, len(ids), 300):
            chunk = ids[start:start+300]
            holders = ",".join("?" for _ in chunk)
            stored = {r["proposition_id"]: r for r in db.execute(f"SELECT proposition_id, document_id, claim FROM proposition WHERE proposition_id IN ({holders})", chunk)}
            for pid in chunk:
                if pid not in stored:
                    missing.append(pid)  # Source proposition remains in the final prompt.
                elif stored[pid]["claim"] != ps[pid]["claim"] or stored[pid]["document_id"] != ps[pid]["document_id"]:
                    raise RuntimeError(f"KG/source proposition mismatch: {pid}")
            for r in db.execute(f"""SELECT m.proposition_id,m.entity_node_id,m.surface,
                       e.preferred_surface,e.canonical_concept_id,e.normalization_level,e.mention_count
                       FROM mention m JOIN entity e ON e.entity_node_id=m.entity_node_id
                       WHERE m.proposition_id IN ({holders}) ORDER BY m.proposition_id,m.entity_node_id,m.surface""", chunk):
                eid = r["entity_node_id"]
                entities[eid] = {"name": r["preferred_surface"], "canonical_concept_id": r["canonical_concept_id"],
                                 "normalization_level": r["normalization_level"], "mention_count": int(r["mention_count"] or 1)}
                surfaces = ps[r["proposition_id"]]["entities"].setdefault(eid, [])
                if r["surface"] not in surfaces:
                    surfaces.append(r["surface"])
            # Indexed assertion lookup, not a full binary_relation scan per question.
            triples.extend(dict(r) for r in db.execute(f"""SELECT edge_id,proposition_id,subject_entity_id,
                    relation_surface,object_entity_id FROM binary_relation
                    WHERE assertion_node_id IN (SELECT assertion_node_id FROM mention
                    WHERE proposition_id IN ({holders})) ORDER BY edge_id""", chunk))
    return {"propositions": ps, "entities": entities, "source_triples": triples,
            "missing_kg_propositions": missing,
            "common_edges": [{"source": p["document_id"], "target": pid, "type": "CONTAINS"} for pid, p in sorted(ps.items())]
                + [{"source": pid, "target": eid, "type": "MENTIONS"} for pid, p in sorted(ps.items()) for eid in sorted(p["entities"])]}


def metadata_identity(a):
    files = {"source_cohort": a.source_cohort, "graph": a.graph,
             "llama_config": a.model / "config.json", "llama_tokenizer": a.model / "tokenizer.json",
             "llama_template": a.model / "tokenizer_config.json", "kev_config": a.kev_adapter / "adapter_config.json",
             "kev_adapter": a.kev_adapter / "adapter_model.safetensors", "kev_head": a.kev_adapter / "head.pt",
             "kev_tokenizer": a.kev_adapter / "tokenizer.json", "kev_base_config": a.kev_base / "config.json",
             "kev_model_code": a.kev_source / "kev/model.py", "kev_api_code": a.kev_source / "kev/api.py",
             "kev_loader_code": a.kev_source / "kev/checkpoint.py"}
    for name, folder in (("llama", a.model), ("kev_base", a.kev_base)):
        shards = sorted(folder.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No model weight shards in {folder}")
        files.update({f"{name}_weight_{i}": path for i, path in enumerate(shards)})
    if a.ontology:
        files["ontology"] = a.ontology
    return {key: identity(path, full_hash=key == "source_cohort") for key, path in files.items()}


def preflight(a):
    inputs = metadata_identity(a)  # Fail before tokenizers or GPU loading.
    rows, gold, benchmark_sources = selected_cohort(a)
    index = build_index(rows, a.graph)
    ontology = g.validate_ontology(json.loads(a.ontology.read_text()) if a.ontology else None, set(index["entities"]))
    ac_jobs = g.annotation_requests(index, a.entities_per_proposition)
    pair_ids = sorted({pair for r in rows for pair in g.pair_candidates(r, index, a.pair_neighbors)})
    b_jobs = [g.claim_request(x, y, index) for x, y in pair_ids]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    lengths = [len(tokenizer.encode(g.render_prompt(tokenizer, r), add_special_tokens=False)) for r in rows]
    if max(lengths) + a.annotation_tokens + a.max_new_tokens > a.max_model_len:
        raise RuntimeError(f"Original evidence does not fit unchanged: max_base_prompt={max(lengths)}, "
                           f"annotation reserve={a.annotation_tokens}, output reserve={a.max_new_tokens}. No evidence was pruned.")
    # Validate the actual local KEV request/encoding API on every bounded request.
    sys.path.insert(0, str(a.kev_source))
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode
    kev_tokenizer = AutoTokenizer.from_pretrained(a.kev_adapter, local_files_only=True)
    max_kev = 0
    for job in ac_jobs + b_jobs:
        record, _ = to_record(SystemOneRequest.model_validate(job["body"]))
        enc = encode(kev_tokenizer, record, max_state=KEV_MAX_STATE, max_branch=KEV_MAX_ROW, strict=True)
        max_kev = max(max_kev, enc["state_tokens"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    contract = {"version": g.VERSION, "hypothesis": "Decision-labeled fixed-evidence graphs improve final reasoning over the identical proposition-only input.",
                "scope": "Exploratory TEST subset, not independent final evaluation; no test-based tuning or automatic scaling.",
                "cohort": {"population": 6545, "sample_ids": [r["sample_id"] for r in rows], "seed": a.seed,
                           "sampling": "dataset-proportional hash sample, no correctness filtering"},
                "inputs": inputs, "benchmark": benchmark_sources, "settings": settings(a),
                "evaluation_labels_sha256": digest(gold),
                "graph_A_variant": "ABOUT+verified_ontology" if a.ontology else "ABOUT_only_no_hierarchy",
                "generation": {"arms": ["text", "graphs"], "backend": "vllm", "temperature": 0,
                    "seed": a.seed, "max_model_len": a.max_model_len, "max_new_tokens": a.max_new_tokens,
                    "annotation_tokens": a.annotation_tokens, "output": "same constrained JSON rationale+answer",
                    "input_order": "question-options-evidence-optional_annotations", "evidence_pruning": False},
                "runtime": {"kev_batch_rows": a.kev_batch_rows, "generation_batch": a.generation_batch,
                            "gpu_memory_utilization": a.gpu_memory_utilization, "physical_gpu": 1,
                            "kev_max_state": KEV_MAX_STATE, "kev_max_row": KEV_MAX_ROW},
                "evaluation": {"primary": "paired MCQ accuracy difference (percentage points)",
                    "missing_parse": "count as incorrect; report separately", "minimum_worthwhile_gain_pp": 3.125,
                    "stop": "No full TEST expansion on this exploratory result alone; inspect labels and confirm on independent validation.",
                    "attribution": "joint graph annotation + expert effect; not isolated topology causality"},
                "code_commit": commit, "code": {p.name: identity(p, full_hash=True) for p in
                    (Path(__file__), ROOT / "medrag/evidence_relation_graph.py", ROOT / "medrag/evidence_graph_runtime.py",
                     ROOT / "medrag/training/graph_input_data.py", ROOT / "medrag/training/graph_input_progress.py")},
                "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "peft")}}
    expected = {"contract": contract, "contract_sha256": digest(contract)}
    path = a.output_root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != expected:
        raise RuntimeError(f"Contract changed; use a new output root, preserve {path}")
    info = {"questions": len(rows), "datasets": dict(Counter(r["dataset"] for r in rows)),
            "unique_documents": len({d["document_id"] for r in rows for d in r["documents"]}),
            "unique_propositions": len(index["propositions"]), "missing_kg_propositions": len(index["missing_kg_propositions"]),
            "empty_document_slots": sum(not d["propositions"] for r in rows for d in r["documents"]),
            "A_C_requests": len(ac_jobs), "B_requests": len(b_jobs),
            "static_decisions": sum(len(j["body"]["questions"]) for j in ac_jobs + b_jobs),
            "max_llama_base_prompt_tokens": max(lengths), "max_kev_state_tokens": max_kev,
            "hierarchy": contract["graph_A_variant"], "gpu_loaded": False, "contract_sha256": digest(contract)}
    return rows, gold, index, ontology, ac_jobs, b_jobs, contract, info


def read_prepared(a):
    manifest = json.loads((a.output_root / "manifest.json").read_text())
    rows = list(jsonl(a.output_root / "cohort.jsonl"))
    return manifest, rows, json.loads((a.output_root / "common_graph.json").read_text())


def kev_worker(a):
    from medrag.evidence_graph_runtime import LocalDecisions
    manifest, rows, index = read_prepared(a)
    c = manifest["contract_sha256"]
    cache = DecisionCache(a.output_root / "decisions", c)
    ac = list(jsonl(a.output_root / "requests_about_role.jsonl"))
    pairs = list(jsonl(a.output_root / "requests_claim.jsonl"))
    emit("stage", index=1, total=1)
    judge = LocalDecisions(a.kev_source, a.kev_adapter, a.kev_base, cache, a.kev_batch_rows)
    emit("update", amount=1)
    emit("complete", message="frozen KEV9B, full-row forward; no external API")
    result = judge.score(ac, 2, emit)
    result.update(judge.score(pairs, 3, emit))
    ontology = json.loads((a.output_root / "ontology.json").read_text())
    graphs, query_jobs, per_question = {}, [], {}
    for row in rows:
        sid = row["sample_id"]
        graph = g.graph_for_question(row, index, ac, pairs, result, settings(a), ontology)
        atomic_json(item_path(a.output_root, "graphs", sid), {"contract_hash": c, **graph})
        graphs[sid] = graph
        per_question[sid] = g.query_requests(row, graph, index, a.max_topics)
        query_jobs.extend(per_question[sid])
    result.update(judge.score(query_jobs, 4, emit))
    group_jobs, per_group = [], {}
    for row in rows:
        sid = row["sample_id"]
        groups = g.make_groups(row, graphs[sid], per_question[sid], result, settings(a))
        per_group[sid] = [g.group_request(row, group, index) for group in groups]
        group_jobs.extend(per_group[sid])
    result.update(judge.score(group_jobs, 5, emit))
    for row in rows:
        sid = row["sample_id"]
        notes = g.final_notes(row, graphs[sid], per_group[sid], result, settings(a))
        atomic_json(item_path(a.output_root, "experts", sid), {"contract_hash": c, "sample_id": sid,
                    "question_requests": per_question[sid], "group_requests": per_group[sid], "notes": notes})
    atomic_json(a.output_root / "KEV_RUNTIME.json", {"batch_parity": judge.parity,
                "batch_disabled": judge.batch_disabled, "oom_splits": judge.oom_splits})
    atomic_json(a.output_root / "KEV_DONE.json", {"contract_hash": c, "questions": len(rows)})


def bounded_notes(a, tokenizer, row, notes):
    # Only truncate derived annotations, never the immutable source propositions.
    notes = json.loads(json.dumps(notes))
    base = len(tokenizer.encode(g.render_prompt(tokenizer, row), add_special_tokens=False))
    dropped = 0
    while notes:
        length = len(tokenizer.encode(g.render_prompt(tokenizer, row, notes), add_special_tokens=False))
        if length - base <= a.annotation_tokens and length + a.max_new_tokens <= a.max_model_len:
            break
        if notes.get("claim_relations"):
            notes["claim_relations"].pop()
        elif notes.get("groups"):
            notes["groups"].pop()
        if not notes.get("groups") and not notes.get("claim_relations"):
            notes = {}
        dropped += 1
    return notes, dropped


def generation_worker(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    manifest, rows, _ = read_prepared(a)
    c = manifest["contract_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    jobs = []
    for row in rows:
        expert = json.loads(item_path(a.output_root, "experts", row["sample_id"]).read_text())
        if expert["contract_hash"] != c:
            raise RuntimeError("Expert cache mismatch")
        notes, dropped = bounded_notes(a, tokenizer, row, expert["notes"])
        for arm in ("text", "graphs"):
            path = item_path(a.output_root, "predictions/" + arm, row["sample_id"])
            prompt = g.render_prompt(tokenizer, row, notes if arm == "graphs" else None)
            prompt_hash = digest(prompt)
            if path.exists():
                old = json.loads(path.read_text())
                if old["contract_hash"] != c or old["prompt_hash"] != prompt_hash:
                    raise RuntimeError(f"Prediction cache mismatch: {path}")
                continue
            if len(tokenizer.encode(prompt, add_special_tokens=False)) + a.max_new_tokens > a.max_model_len:
                raise RuntimeError("Exact generation prompt exceeds context; no truncation allowed")
            jobs.append({"row": row, "arm": arm, "prompt": prompt, "prompt_hash": prompt_hash,
                         "path": path, "notes": notes if arm == "graphs" else {}, "dropped": dropped if arm == "graphs" else 0})
    emit("stage", index=6, total=1, initial=0 if jobs else 1)
    if not jobs:
        emit("complete", message="all predictions cached; Llama not loaded")
        emit("stage", index=7, total=2 * len(rows), initial=2 * len(rows), unit="answer")
        emit("complete", message="paired answers cached")
        return
    llm = LLM(model=str(a.model), tokenizer=str(a.model), dtype="bfloat16", trust_remote_code=False,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
              max_num_seqs=a.generation_batch, max_num_batched_tokens=8192,
              enable_prefix_caching=True, seed=a.seed, disable_log_stats=True)
    emit("update", amount=1)
    emit("complete", message="frozen pristine Llama8B; KEV process already exited")
    emit("stage", index=7, total=2 * len(rows), initial=2 * len(rows) - len(jobs), unit="answer", kind="generation")
    # Adjacent paired arms share the evidence prefix. Both use identical JSON grammar and token budgets.
    for start in range(0, len(jobs), a.generation_batch):
        batch = jobs[start:start+a.generation_batch]
        params = [SamplingParams(temperature=0, seed=a.seed, max_tokens=a.max_new_tokens,
                    structured_outputs=StructuredOutputsParams(json=g.response_schema(j["row"]["options"]))) for j in batch]
        before = time.monotonic()
        outputs = llm.generate([j["prompt"] for j in batch], params, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("Llama returned an incomplete batch")
        elapsed = time.monotonic() - before
        for job, out in zip(batch, outputs, strict=True):
            completion = out.outputs[0]
            parsed = g.parse_answer(completion.text, job["row"]["options"])
            atomic_json(job["path"], {"contract_hash": c, "sample_id": job["row"]["sample_id"],
                "dataset": job["row"]["dataset"], "arm": job["arm"], "prompt_hash": job["prompt_hash"],
                "raw": completion.text, **parsed, "finish_reason": completion.finish_reason,
                "prompt_tokens": len(out.prompt_token_ids), "output_tokens": len(completion.token_ids),
                "notes": job["notes"], "dropped_annotations": job["dropped"],
                "batch_seconds": elapsed, "batch_size": len(batch),
                "proposition_ids": [p["proposition_id"] for p in g.propositions(job["row"])]})
            emit("update", amount=1)
    emit("complete", message=f"paired generations durable={a.output_root / 'predictions'}")


def report(a):
    manifest, rows, _ = read_prepared(a)
    c = manifest["contract_sha256"]
    gold = json.loads((a.output_root / "evaluation_labels.json").read_text())
    if digest(gold) != manifest["contract"]["evaluation_labels_sha256"]:
        raise RuntimeError("Evaluation labels changed")
    label_settings = manifest["contract"]["settings"]
    seed = manifest["contract"]["cohort"]["seed"]
    paired, missing, label_counts, graph_counts = [], [], Counter(), Counter()
    for row in rows:
        sid = row["sample_id"]
        result = {}
        for arm in ("text", "graphs"):
            path = item_path(a.output_root, "predictions/" + arm, sid)
            if path.exists():
                data = json.loads(path.read_text())
                if data["contract_hash"] != c:
                    raise RuntimeError("Report prediction contract mismatch")
                result[arm] = data
        if len(result) != 2:
            missing.append(sid)
            continue
        bc, gc = (result[arm]["answer"] == gold[sid] for arm in ("text", "graphs"))
        paired.append({"sample_id": sid, "dataset": row["dataset"], "gold": gold[sid],
                       "text_answer": result["text"]["answer"], "graphs_answer": result["graphs"]["answer"],
                       "text_correct": bc, "graphs_correct": gc, "transition": ("C" if bc else "W") + "->" + ("C" if gc else "W"),
                       "text_parse_failure": result["text"]["parse_error"] is not None,
                       "graphs_parse_failure": result["graphs"]["parse_error"] is not None,
                       "text_output_tokens": result["text"]["output_tokens"], "graphs_output_tokens": result["graphs"]["output_tokens"]})
        gp = json.loads(item_path(a.output_root, "graphs", sid).read_text())
        for name in ("A_topic", "B_claim", "C_role"):
            graph_counts[name] += len(gp[name])
        graph_counts["questions_without_notes"] += int(not result["graphs"]["notes"])
    def metric(items):
        n = len(items)
        b, t = sum(x["text_correct"] for x in items), sum(x["graphs_correct"] for x in items)
        return {"N": n, "text_correct": b, "graphs_correct": t,
                "text_accuracy_pct": 100*b/n if n else None, "graphs_accuracy_pct": 100*t/n if n else None,
                "delta_pp": 100*(t-b)/n if n else None, "transitions": dict(Counter(x["transition"] for x in items)),
                "text_parse_failures": sum(x["text_parse_failure"] for x in items),
                "graphs_parse_failures": sum(x["graphs_parse_failure"] for x in items)}
    overall = metric(paired)
    if paired:
        rng = random.Random(seed)
        deltas = [int(x["graphs_correct"]) - int(x["text_correct"]) for x in paired]
        bootstrap = sorted(100*sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(2000))
        overall["paired_bootstrap_95ci_delta_pp"] = [bootstrap[49], bootstrap[1949]]
    subsets = {d: metric([x for x in paired if x["dataset"] == d]) for d in sorted({x["dataset"] for x in paired})}
    pooled = {d: metric([x for x in paired if ("mmlu" if x["dataset"].startswith("mmlu_") else x["dataset"]) == d])
              for d in ("medmcqa", "medqa", "mmlu")}
    # Blind annotation audit export: model predictions, inputs and blank human labels; never report agreement as accuracy.
    audits = []
    unique_requests = 0
    for path in sorted((a.output_root / "decisions").glob("*/*.json")):
        value = json.loads(path.read_text())
        if value["contract_hash"] != c:
            raise RuntimeError("Annotation report cache contract mismatch")
        unique_requests += 1
        for key, answer in value["answers"].items():
            label = g.decision(answer, label_settings["confidence"], label_settings["margin"])
            task = value["request"]["task"]
            label_counts[f"{task}:{label}"] += 1
            audits.append({"request_id": value["request"]["id"], "task": task, "decision_key": key,
                           "state": value["request"]["body"]["state"],
                           "question": value["request"]["body"]["questions"][key],
                           "model_label": label, "probabilities": answer["probabilities"], "human_label": None})
    strata = {}
    for audit in sorted(audits, key=lambda x: digest([seed, x["request_id"], x["decision_key"]])):
        key = (audit["task"], audit["model_label"])
        strata.setdefault(key, [])
        if len(strata[key]) < 4:
            strata[key].append(audit)
    summary = {"contract_hash": c, "requested_N": len(rows), "paired_completed_N": len(paired),
               "missing_sample_ids": missing, "overall": overall, "by_dataset": subsets, "pooled": pooled,
               "graph_edge_counts_over_questions": dict(graph_counts), "decision_label_counts": dict(label_counts),
               "cost": {"unique_kev_requests": unique_requests, "kev_decisions": sum(label_counts.values()),
                        "paired_llama_generations": 2*len(paired),
                        "mean_text_output_tokens": sum(x["text_output_tokens"] for x in paired)/len(paired) if paired else None,
                        "mean_graph_output_tokens": sum(x["graphs_output_tokens"] for x in paired)/len(paired) if paired else None,
                        "stage_wall_times": str(a.output_root / "RUNTIME.json")},
               "hierarchy": manifest["contract"]["graph_A_variant"],
               "annotation_accuracy": "not measured; human labels required in annotation_audit.jsonl",
               "minimum_worthwhile_gain_pp": 3.125,
               "observed_gain_threshold_met": not missing and bool(paired) and overall["delta_pp"] >= 3.125,
               "scale_authorized": False,
               "caveat": "Exploratory TEST sample. Paired effect includes annotation/decision work, not isolated graph topology. Do not tune on this sample."}
    if all(x["N"] for x in pooled.values()):
        summary["macro3"] = {k: sum(x[k] for x in pooled.values())/3 for k in ("text_accuracy_pct", "graphs_accuracy_pct", "delta_pp")}
    atomic_jsonl(a.output_root / "paired_predictions.jsonl", paired)
    atomic_jsonl(a.output_root / "annotation_audit.jsonl", [v for values in strata.values() for v in values])
    atomic_json(a.output_root / "summary.json", summary)
    lines = ["# Exploratory paired TEST pilot", "", f"Completed {len(paired)}/{len(rows)}; missing {len(missing)}.", "",
             "|Subset|N|Text accuracy|Graphs accuracy|Delta pp|C→W|W→C|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in [("overall", overall), *pooled.items()]:
        if m["N"]:
            lines.append(f"|{name}|{m['N']}|{m['text_accuracy_pct']:.2f}|{m['graphs_accuracy_pct']:.2f}|{m['delta_pp']:+.2f}|{m['transitions'].get('C->W',0)}|{m['transitions'].get('W->C',0)}|")
    lines += ["", summary["caveat"], "", "KEV label correctness is NOT automatically verified. Review annotation_audit.jsonl."]
    # This is a generated run artifact, not repository documentation.
    temp = a.output_root / "summary.md.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(a.output_root / "summary.md")
    return summary


def child(a, worker, progress):
    argv = [x for x in sys.argv[1:] if x != "--run"]
    command = [sys.executable, str(Path(__file__)), "--worker", worker, *argv]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=os.environ.copy())
    try:
        for line in process.stdout:
            if not line.startswith(EVENT):
                if line.strip() and "Loading weights:" not in line:
                    progress.log(line.rstrip())
                continue
            data = json.loads(line[len(EVENT):])
            event = data.pop("event")
            if event == "stage":
                progress.stage(**data)
            elif event == "update":
                progress.update(**data)
            elif event == "complete":
                progress.complete(**data)
        if process.wait() != 0:
            raise RuntimeError(f"{worker} worker failed (exit={process.returncode}); existing shards are preserved")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise


def run(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Use the launcher: physical GPU1 only")
    print("[plan] proportional TEST sample -> fixed source/KG -> KEV A/C+B -> decision experts -> unload KEV -> paired Llama -> report", flush=True)
    print("[scope] no training/retrieval changes; exact source text retained; exploratory TEST only. No hierarchy without --ontology.", flush=True)
    print("[ETA] new labeling workloads have no matched throughput; stage ETA calibrates on fresh decisions; generation separately. Cached work never inflates rate.", flush=True)
    progress = PilotProgress(STAGES)
    try:
        progress.stage(0, 1)
        rows, gold, index, ontology, ac, pairs, contract, info = preflight(a)
        c = ensure_contract(a.output_root / "manifest.json", contract)
        progress.bind(a.output_root, c)
        atomic_jsonl(a.output_root / "cohort.jsonl", rows)
        atomic_json(a.output_root / "evaluation_labels.json", gold)
        atomic_json(a.output_root / "common_graph.json", index)
        atomic_json(a.output_root / "ontology.json", ontology)
        atomic_jsonl(a.output_root / "requests_about_role.jsonl", ac)
        atomic_jsonl(a.output_root / "requests_claim.jsonl", pairs)
        atomic_json(a.output_root / "PREFLIGHT.json", info)
        progress.update()
        progress.complete(json.dumps(info))
        done = a.output_root / "KEV_DONE.json"
        if done.exists():
            if json.loads(done.read_text())["contract_hash"] != c:
                raise RuntimeError("KEV completion contract mismatch")
            for i in range(1, 6):
                progress.stage(i, 1, initial=1)
                progress.complete("complete compatible KEV stage cached; model not loaded")
        else:
            child(a, "kev", progress)
        child(a, "generate", progress)
        progress.stage(8, 1)
        summary = report(a)
        progress.update()
        progress.complete(str(a.output_root / "summary.json"))
        print("[result] " + json.dumps(summary["overall"], ensure_ascii=False), flush=True)
    except BaseException as error:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(STAGES)} item={progress.done}/{progress.total}; "
                     f"remaining={progress.total-progress.done}; {type(error).__name__}: {error}; "
                     f"durable={a.output_root}; identical command resumes matching decision/prediction shards")
        raise
    finally:
        progress.close()


def main():
    a = arguments()
    warnings.filterwarnings("ignore", message="Unexpected keyword arguments.*", category=UserWarning)
    if a.worker:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("Workers require physical GPU1")
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()
        (kev_worker if a.worker == "kev" else generation_worker)(a)
    elif a.preflight:
        print(json.dumps(preflight(a)[-1], ensure_ascii=False, indent=2))
    elif a.report:
        print(json.dumps(report(a)["overall"], ensure_ascii=False, indent=2))
    else:
        run(a)


if __name__ == "__main__":
    main()

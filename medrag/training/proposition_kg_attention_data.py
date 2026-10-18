"""Official-train, exact-text layouts for whole-proposition attention.

No retrieval, LLM, GPU, or writes occur here. Original source text is retained;
the graph supplies only existing endpoint identities and subject/object roles.
The main runner owns versioned manifests and atomic cache writes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from medrag.training.graph_input_data import SYSTEM, digest, identity, jsonl, sha_file

VERSION = "official_train_proposition_attention_data_v1"
ROLE_NAMES = ("SS", "SO", "OS", "OO")  # receiver role, then sender role
SOURCE_FILES = ("COMPLETE.json", "manifest.json", "question_views.jsonl",
                "cohort/labels.jsonl", "cohort/COMPLETE.json",
                "usable/usable_propositions.jsonl", "usable/COMPLETE.json",
                "kg/graph.sqlite3", "kg/COMPLETE.json")


def inspect_inputs(root, model):
    """Cheap metadata-only preflight; deliberately no tokenizer or data scan."""
    root, model = Path(root).resolve(), Path(model).resolve()
    for rel in SOURCE_FILES:
        if not (root / rel).is_file() or not (root / rel).stat().st_size:
            raise FileNotFoundError(root / rel)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if (complete.get("version") != "official_train10000_top8_kg_v1"
            or complete.get("questions") != 10000
            or complete.get("split_counts") != {"train": 9000, "validation": 1000}):
        raise ValueError("Expected completed official TRAIN 9000/1000 cohort, not TEST re-split")
    for rel in ("config.json", "tokenizer.json", "tokenizer_config.json",
                "model.safetensors.index.json"):
        if not (model / rel).is_file():
            raise FileNotFoundError(model / rel)
    config = json.loads((model / "config.json").read_text())
    if (config.get("model_type") != "llama" or config.get("num_hidden_layers") != 32
            or config.get("hidden_size") != 4096
            or config.get("max_position_embeddings", 0) < 8192):
        raise ValueError("This experiment requires the configured pristine 32-layer Llama-3-8B")
    if (model / "adapter_config.json").exists():
        raise ValueError("Do not reuse the TEST-trained medical LoRA/adapter")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("Empty Llama weight index")
    for name in shard_names:
        shard = (model / name).resolve()
        if shard.parent != model or not shard.is_file() or not shard.stat().st_size:
            raise FileNotFoundError(f"Missing/invalid model shard: {shard}")
    # Schema inspection does not scan the graph and the connection is read-only.
    with sqlite3.connect(f"file:{root / 'kg/graph.sqlite3'}?mode=ro", uri=True) as con:
        fields = {row[1] for row in con.execute("PRAGMA table_info(binary_relation)")}
        required = {"edge_id", "proposition_id", "document_id",
                    "subject_entity_id", "object_entity_id"}
        if not required <= fields:
            raise ValueError("Missing required source KG fields")
    return {"version": VERSION, "source_root": str(root), "model": str(model),
            "source_completion": complete,
            "sources": {rel: identity(root / rel) for rel in SOURCE_FILES},
            "model_files": {rel: identity(model / rel) for rel in
                            ["config.json", "tokenizer.json", "tokenizer_config.json",
                             "model.safetensors.index.json", *shard_names]},
            "source_disk_free_bytes": shutil.disk_usage(root).free,
            "no_gpu_loaded": True, "large_file_checks": "size/mtime only until materialization",
            "prompt_order": "Evidence -> question stem -> options -> assistant Answer:",
            "controller_features": "proposition spans and question-stem span only",
            "proposition_order": "retrieved document rank, then source JSONL encounter order",
            "candidate_rule": "shared existing binary-relation endpoint ID; receiver > sender",
            "role_names": list(ROLE_NAMES)}


def _progress(callback, completed, total, phase):
    if callback is not None:
        callback(completed, total, phase)


def render_prompt(tokenizer, question, options, propositions):
    """Return text plus character spans, excluding wrappers, choices and answers."""
    user = "Evidence:\n"
    prop_spans = []
    if not propositions:
        user += "(No supplied evidence.)\n"
    for i, text in enumerate(propositions):
        user += f"[P{i + 1}] "
        start = len(user)
        user += text
        prop_spans.append((start, len(user)))
        user += "\n\n"
    user += "\nQuestion:\n"
    q_start = len(user)
    user += question
    question_span = (q_start, len(user))
    user += "\n\nOptions:\n" + "\n".join(f"{key}. {options[key]}" for key in sorted(options))
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True) + "Answer:"
    # Search for the complete user message, not individual repeated sentences.
    offset = prompt.find(user)
    if offset < 0 or prompt.find(user, offset + 1) >= 0:
        raise ValueError("Chat template must preserve exactly one complete user message")
    return (prompt, [(a + offset, b + offset) for a, b in prop_spans],
            (question_span[0] + offset, question_span[1] + offset))


def _token_span(offsets, char_span, prompt):
    a, b = char_span
    selected = [i for i, (start, end) in enumerate(offsets)
                if end > start and start < b and end > a]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("Missing or non-contiguous exact source token span")
    left, right = offsets[selected[0]][0], offsets[selected[-1]][1]
    # BPE commonly includes the separating leading space in the first word.
    # Only whitespace may cross a source boundary, never labels or options.
    if (left < a and prompt[left:a].strip()) or (right > b and prompt[b:right].strip()):
        raise ValueError("Source span overlaps a token outside its text boundary")
    return [selected[0], selected[-1] + 1]


def make_layout(tokenizer, view, label, props, endpoint_roles, max_prompt_tokens=8064):
    """Build one exact layout; trailing whole propositions are the only truncation."""
    if not 1 <= max_prompt_tokens <= 8064:
        raise ValueError("max_prompt_tokens must reserve 128 tokens within the 8192 context")
    options = view["options"]
    labels = sorted(options)
    if (not view["question"].strip() or not labels or label not in labels
            or any(not isinstance(v, str) or not v.strip() for v in options.values())):
        raise ValueError(f"Invalid stem/options/answer: {view['sample_id']}")

    def encode(n):
        prompt, pchars, qchars = render_prompt(tokenizer, view["question"], options,
                                             [p["claim"] for p in props[:n]])
        encoding = tokenizer(prompt, add_special_tokens=False, truncation=False,
                             return_offsets_mapping=True)
        return prompt, pchars, qchars, encoding

    # Do not assume token length is monotone around the empty-evidence wrapper.
    prompt, pchars, qchars, encoded = encode(len(props))
    original_tokens = len(encoded["input_ids"])
    kept = len(props)
    while len(encoded["input_ids"]) > max_prompt_tokens and kept:
        kept -= 1
        prompt, pchars, qchars, encoded = encode(kept)
    if len(encoded["input_ids"]) > max_prompt_tokens:
        raise ValueError(f"Stem/options alone exceed context: {view['sample_id']}")
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    spans = [_token_span(offsets, span, prompt) for span in pchars]
    qspan = _token_span(offsets, qchars, prompt)
    if any(end > qspan[0] for _, end in spans):
        raise ValueError("Evidence must precede question, and options must follow question")
    choice_ids = []
    for option in labels:
        full = tokenizer(prompt + " " + option, add_special_tokens=False,
                         truncation=False)["input_ids"]
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise ValueError(f"Choice {option!r} is not one stable token after Answer:")
        choice_ids.append(full[-1])
    if len(set(choice_ids)) != len(choice_ids):
        raise ValueError("Distinct option labels map to the same output token")
    retained = props[:kept]
    posting = defaultdict(list)
    for i, prop in enumerate(retained):
        for entity, roles in endpoint_roles.get(prop["proposition_id"], {}).items():
            posting[entity].append((i, roles))
    pair_roles = {}
    for values in posting.values():
        for recv, rr in values:
            for sender, sr in values:
                if recv <= sender:
                    continue
                flags = pair_roles.setdefault((recv, sender), [0.0] * 4)
                for r in rr:
                    for s in sr:
                        flags[r * 2 + s] = 1.0
    pairs = sorted(pair_roles)
    linked = {index for pair in pairs for index in pair}
    return {"sample_id": view["sample_id"], "dataset": view["dataset"],
            "split": view["split"], "input_ids": ids,
            "proposition_spans": spans, "question_span": qspan,
            "pairs": [list(pair) for pair in pairs],
            "pair_roles": [pair_roles[pair] for pair in pairs],
            "option_labels": labels, "choice_token_ids": choice_ids,
            "gold_index": labels.index(label),
            "proposition_ids": [p["proposition_id"] for p in retained],
            "proposition_document_ids": [p["document_id"] for p in retained],
            "source_document_ids": list(view["document_ids"]),
            "graph_stats": {"propositions": kept, "causal_pairs": len(pairs),
                            "isolated_propositions": kept - len(linked)},
            "truncation": {"original_propositions": len(props),
                           "removed_propositions": len(props) - kept,
                           "original_prompt_tokens": original_tokens,
                           "prompt_tokens": len(ids)}}


def build_rows(root, tokenizer, max_prompt_tokens=8064, progress_callback=None):
    """Materialize official train/validation only, with complete source checks.

    ``progress_callback(completed, total, phase)`` is optional. This function
    writes nothing and does not load model weights. The runner caches its result.
    """
    root = Path(root)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if sha_file(root / "question_views.jsonl") != complete["question_views_sha256"]:
        raise ValueError("Official question-view completion hash mismatch")
    views = list(jsonl(root / "question_views.jsonl"))
    if len(views) != complete["questions"]:
        raise ValueError("Question count differs from completed dataset")
    if Counter(v["split"] for v in views) != complete["split_counts"]:
        raise ValueError("Official train/validation counts changed")
    ids = [v["sample_id"] for v in views]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question ID")
    labels = {}
    for row in jsonl(root / "cohort/labels.jsonl"):
        if row["sample_id"] in labels:
            raise ValueError("Duplicate label ID")
        labels[row["sample_id"]] = row["answer"]
    if set(labels) != set(ids):
        raise ValueError("Question/label ID mismatch")
    stems = {}
    for view in views:
        if (view["split"] not in ("train", "validation")
                or view.get("official_split") != "train"
                or view["dataset"] not in ("medmcqa", "medqa")):
            raise ValueError("Non-official-train question in training inputs")
        stem = " ".join(view["question"].casefold().split())
        if stem in stems and stems[stem] != view["split"]:
            raise ValueError("Exact question stem crosses train/validation splits")
        stems[stem] = view["split"]
        if len(view["document_ids"]) != len(set(view["document_ids"])) or len(view["document_ids"]) > 8:
            raise ValueError("Expected unique fixed Top-8 documents")
        if len(view["usable_proposition_ids"]) != len(set(view["usable_proposition_ids"])):
            raise ValueError("Duplicate proposition in question view")
    _progress(progress_callback, len(views), len(views), "load_views")
    wanted_props = {pid for v in views for pid in v["usable_proposition_ids"]}
    props = {}
    for order, prop in enumerate(jsonl(root / "usable/usable_propositions.jsonl")):
        pid = prop["proposition_id"]
        if pid not in wanted_props:
            continue
        if pid in props or not prop["claim"].strip():
            raise ValueError("Duplicate or empty usable source proposition")
        props[pid] = {"proposition_id": pid, "document_id": prop["document_id"],
                      "claim": prop["claim"], "source_order": order}
        if len(props) % 10000 == 0:
            _progress(progress_callback, len(props), len(wanted_props), "load_propositions")
    if set(props) != wanted_props:
        raise ValueError("Question views reference missing usable propositions")
    _progress(progress_callback, len(props), len(wanted_props), "load_propositions")
    wanted_edges = {eid for v in views for eid in v["binary_edge_ids"]}
    edges = {}
    with sqlite3.connect(f"file:{root / 'kg/graph.sqlite3'}?mode=ro", uri=True) as con:
        for eid, pid, doc, subject, obj in con.execute(
                "SELECT edge_id,proposition_id,document_id,subject_entity_id,object_entity_id FROM binary_relation"):
            if eid in wanted_edges:
                if pid not in props or not subject or not obj:
                    raise ValueError("Invalid/missing source endpoint identity")
                edges[eid] = (pid, doc, subject, obj)
                if len(edges) % 10000 == 0:
                    _progress(progress_callback, len(edges), len(wanted_edges), "load_edges")
    if set(edges) != wanted_edges:
        raise ValueError("Question views reference missing KG edges")
    _progress(progress_callback, len(edges), len(wanted_edges), "load_edges")
    rows = []
    for count, view in enumerate(views, 1):
        rank = {doc: i for i, doc in enumerate(view["document_ids"])}
        ordered = [props[pid] for pid in view["usable_proposition_ids"]]
        if any(p["document_id"] not in rank for p in ordered):
            raise ValueError("Proposition source lies outside the fixed Top-8")
        ordered.sort(key=lambda p: (rank[p["document_id"]], p["source_order"]))
        allowed = set(view["usable_proposition_ids"])
        roles = defaultdict(lambda: defaultdict(set))
        for eid in view["binary_edge_ids"]:
            pid, doc, subject, obj = edges[eid]
            if pid not in allowed or doc not in rank or props[pid]["document_id"] != doc:
                raise ValueError("KG provenance lies outside question evidence")
            roles[pid][subject].add(0)
            roles[pid][obj].add(1)
        rows.append(make_layout(tokenizer, view, labels[view["sample_id"]], ordered,
                                roles, max_prompt_tokens))
        _progress(progress_callback, count, len(views), "tokenize")
    summary = {"version": VERSION, "questions": len(rows),
               "split_counts": dict(Counter(r["split"] for r in rows)),
               "by_dataset": {split: dict(Counter(r["dataset"] for r in rows if r["split"] == split))
                              for split in ("train", "validation")},
               "max_prompt_tokens": max(len(r["input_ids"]) for r in rows),
               "empty_evidence_questions": sum(not r["proposition_spans"] for r in rows),
               "zero_pair_questions": sum(not r["pairs"] for r in rows),
               "total_propositions": sum(len(r["proposition_spans"]) for r in rows),
               "total_causal_pairs": sum(len(r["pairs"]) for r in rows),
               "truncated_questions": sum(r["truncation"]["removed_propositions"] > 0 for r in rows),
               "removed_propositions": sum(r["truncation"]["removed_propositions"] for r in rows),
               "sample_ids_sha256": digest(ids), "question_views_sha256": complete["question_views_sha256"],
               "answer_or_options_in_controller_features": False,
               "proposition_text_changed": False}
    return rows, summary


def shuffle_pairs(row, seed):
    """Directed degree-preserving swaps within the causal lower triangle.

    Source/receiver degrees, pair count, role-vector multiset, and text are kept.
    Connectivity/components need not be preserved. Some ordered graphs admit no
    valid swap; report this rather than claiming an effective randomization.
    """
    result = copy.deepcopy(row)
    pairs = [tuple(pair) for pair in row["pairs"]]
    roles = [list(role) for role in row["pair_roles"]]
    if len(pairs) != len(roles) or len(set(pairs)) != len(pairs) or any(a <= b for a, b in pairs):
        raise ValueError("Invalid causal pair layout for shuffle")
    original, present = set(pairs), set(pairs)
    stable_seed = int(hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()[:16], 16)
    rng = random.Random(stable_seed)
    accepted = 0
    if len(pairs) >= 2:
        for _ in range(max(100, 20 * len(pairs))):
            i, j = rng.sample(range(len(pairs)), 2)
            a, b = pairs[i]
            c, d = pairs[j]
            x, y = (a, d), (c, b)
            if a == c or b == d or a <= d or c <= b or x in present or y in present:
                continue
            present.remove(pairs[i]); present.remove(pairs[j])
            present.update((x, y)); pairs[i], pairs[j] = x, y
            accepted += 1
    ordered = sorted(zip(pairs, roles), key=lambda item: item[0])
    result["pairs"] = [list(pair) for pair, _ in ordered]
    result["pair_roles"] = [role for _, role in ordered]
    changed = len(present - original)
    result["shuffle_statistics"] = {"seed": seed, "accepted_swaps": accepted,
                         "changed_pairs": changed, "pairs": len(pairs),
                         "changed_fraction": changed / len(pairs) if pairs else 0.0,
                         "unshufflable_or_unchanged": changed == 0,
                         "preserves": "directed degrees, causal mask, pair count, role multiset, text"}
    return result

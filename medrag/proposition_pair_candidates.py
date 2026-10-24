"""Question-independent sparse pair proposals, restricted to fixed Top-8 co-occurrence.

These are candidates for independent A/B/C decisions, never semantic edges.
Question text, options, answers and KEV are deliberately absent.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import sqlite3

from medrag.evidence_relation_graph import GENERIC
from medrag.training.graph_input_data import digest, jsonl

VERSION = "common_proposition_pairs_v1"
CHANNELS = ("shared_entity", "direct_kg", "semantic", "document_adjacent")


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA synchronous=FULL")
    return db


def initialize(db, contract_hash):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS document(did TEXT PRIMARY KEY,signature TEXT NOT NULL,n INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS proposition(
          idx INTEGER PRIMARY KEY,pid TEXT UNIQUE NOT NULL,did TEXT NOT NULL,pos INTEGER NOT NULL,
          claim TEXT NOT NULL,kg_present INTEGER,UNIQUE(did,pos));
        CREATE INDEX IF NOT EXISTS prop_doc ON proposition(did,pos);
        CREATE TABLE IF NOT EXISTS question(qid TEXT PRIMARY KEY,dataset TEXT NOT NULL,ordinal INTEGER UNIQUE);
        CREATE TABLE IF NOT EXISTS question_document(
          qid TEXT NOT NULL,did TEXT NOT NULL,rank INTEGER NOT NULL,PRIMARY KEY(qid,rank),UNIQUE(qid,did));
        CREATE TABLE IF NOT EXISTS entity(
          eid TEXT PRIMARY KEY,name TEXT NOT NULL,canonical TEXT,level TEXT,doc_frequency INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS prop_entity(idx INTEGER,eid TEXT,PRIMARY KEY(idx,eid));
        CREATE INDEX IF NOT EXISTS pe_entity ON prop_entity(eid,idx);
        CREATE TABLE IF NOT EXISTS triple(
          tid TEXT PRIMARY KEY,idx INTEGER,s TEXT,r TEXT,o TEXT);
        CREATE INDEX IF NOT EXISTS triple_prop ON triple(idx);
        CREATE TABLE IF NOT EXISTS pair(
          a INTEGER,b INTEGER,PRIMARY KEY(a,b),CHECK(a<b));
        CREATE TABLE IF NOT EXISTS question_result(qid TEXT PRIMARY KEY,body TEXT NOT NULL);
    """)
    old = get_meta(db, "contract_hash")
    if old is not None and old != contract_hash:
        raise RuntimeError("Candidate database contract mismatch; use a new output root")
    set_meta(db, "contract_hash", contract_hash)
    db.commit()


def get_meta(db, key, default=None):
    r = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))


def ingest(db, source, expected_questions=6545, expected_propositions=283462, tick=lambda: None):
    """Atomic CPU inventory. Keep source IDs, original claims and list order exactly."""
    if get_meta(db, "inventory_done"):
        return get_meta(db, "inventory_stats")
    slots = empty = occurrences = total_pairs = 0
    doc_pairs, seen_questions = set(), set()
    known_docs, known_props = {}, set()
    with db:
        for ordinal, row in enumerate(jsonl(source)):
            qid = row["sample_id"]
            if qid in seen_questions:
                raise ValueError(f"Duplicate question ID: {qid}")
            seen_questions.add(qid)
            documents = row["documents"]
            dids = [d["document_id"] for d in documents]
            if len(documents) != 8 or len(set(dids)) != 8 or [d["rank"] for d in documents] != list(range(1, 9)):
                raise ValueError(f"Expected unchanged, unique ranked Top-8: {qid}")
            db.execute("INSERT INTO question VALUES (?,?,?)", (qid, row["dataset"], ordinal))
            qps = set()
            for d in documents:
                did = d["document_id"]
                ps = d["propositions"]
                signature = digest([(p["proposition_id"], p["claim"]) for p in ps])
                slots += 1
                empty += not ps
                db.execute("INSERT INTO question_document VALUES (?,?,?)", (qid, did, d["rank"]))
                if did in known_docs:
                    if known_docs[did] != (signature, len(ps)):
                        raise ValueError(f"Document text/order changed across questions: {did}")
                else:
                    known_docs[did] = (signature, len(ps))
                    db.execute("INSERT INTO document VALUES (?,?,?)", (did, signature, len(ps)))
                    for pos, p in enumerate(ps):
                        pid, claim = p["proposition_id"], p["claim"]
                        if pid in known_props or not isinstance(claim, str) or not claim.strip():
                            raise ValueError(f"Duplicate/invalid proposition identity: {pid}")
                        idx = len(known_props)
                        known_props.add(pid)
                        db.execute("INSERT INTO proposition(idx,pid,did,pos,claim) VALUES (?,?,?,?,?)",
                                   (idx, pid, did, pos, claim))
                for p in ps:
                    if p["proposition_id"] in qps:
                        raise ValueError(f"Duplicate proposition within Top-8: {qid}")
                    qps.add(p["proposition_id"])
            doc_pairs.update(itertools.combinations(sorted(dids), 2))
            occurrences += len(qps)
            total_pairs += len(qps) * (len(qps)-1) // 2
            tick()
        if len(seen_questions) != expected_questions or len(known_props) != expected_propositions:
            raise ValueError(f"Source contract changed: questions={len(seen_questions)}, propositions={len(known_props)}")
        unique_pairs = sum(n*(n-1)//2 for _, n in known_docs.values())
        unique_pairs += sum(known_docs[a][1]*known_docs[b][1] for a, b in doc_pairs)
        stats = dict(questions=len(seen_questions), document_slots=slots, empty_document_slots=empty,
                     unique_documents=len(known_docs), unique_propositions=len(known_props),
                     proposition_occurrences=occurrences, within_question_all_pair_occurrences=total_pairs,
                     unique_cooccurring_pairs=unique_pairs)
        set_meta(db, "inventory_stats", stats)
        set_meta(db, "inventory_done", True)
    return stats


def import_kg(db, graph_path, tick=lambda n: None, chunk_size=300):
    """Indexed source reads; resume after each committed proposition block."""
    if get_meta(db, "kg_done"):
        return
    source = sqlite3.connect(f"file:{Path(graph_path).resolve()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        cursor = get_meta(db, "kg_next", 0)
        while True:
            ps = list(db.execute("SELECT idx,pid,did,claim FROM proposition WHERE idx>=? ORDER BY idx LIMIT ?",
                                 (cursor, chunk_size)))
            if not ps:
                break
            ids = [p["pid"] for p in ps]
            lookup = {p["pid"]: p["idx"] for p in ps}
            marks = ",".join("?" for _ in ids)
            stored = {p["proposition_id"]: p for p in source.execute(
                f"SELECT proposition_id,document_id,claim FROM proposition WHERE proposition_id IN ({marks})", ids)}
            with db:
                for p in ps:
                    original = stored.get(p["pid"])
                    if original and (original["document_id"] != p["did"] or original["claim"] != p["claim"]):
                        raise ValueError(f"Source proposition differs from KG: {p['pid']}")
                    db.execute("UPDATE proposition SET kg_present=? WHERE idx=?", (int(original is not None), p["idx"]))
                for e in source.execute(f"""SELECT DISTINCT m.proposition_id,m.entity_node_id,
                    e.preferred_surface,e.canonical_concept_id,e.normalization_level
                    FROM mention m JOIN entity e ON m.entity_node_id=e.entity_node_id
                    WHERE m.proposition_id IN ({marks})""", ids):
                    db.execute("INSERT OR IGNORE INTO entity(eid,name,canonical,level) VALUES (?,?,?,?)",
                               (e["entity_node_id"], e["preferred_surface"] or "", e["canonical_concept_id"], e["normalization_level"]))
                    db.execute("INSERT OR IGNORE INTO prop_entity VALUES (?,?)", (lookup[e["proposition_id"]], e["entity_node_id"]))
                for t in source.execute(f"""SELECT edge_id,proposition_id,subject_entity_id,relation_surface,object_entity_id
                    FROM binary_relation WHERE assertion_node_id IN
                    (SELECT assertion_node_id FROM mention WHERE proposition_id IN ({marks}))""", ids):
                    db.execute("INSERT OR IGNORE INTO triple VALUES (?,?,?,?,?)", (t["edge_id"], lookup[t["proposition_id"]],
                               t["subject_entity_id"], t["relation_surface"], t["object_entity_id"]))
                cursor = ps[-1]["idx"] + 1
                set_meta(db, "kg_next", cursor)
            tick(len(ps))
        with db:
            # Frequency over distinct documents in this immutable retrieved union.
            db.execute("""CREATE TEMP TABLE frequencies AS SELECT e.eid,COUNT(DISTINCT p.did) AS n
                FROM prop_entity e JOIN proposition p ON p.idx=e.idx GROUP BY e.eid""")
            db.execute("CREATE UNIQUE INDEX temp.freq_id ON frequencies(eid)")
            db.execute("UPDATE entity SET doc_frequency=(SELECT n FROM frequencies WHERE eid=entity.eid)")
            db.execute("DROP TABLE frequencies")
            set_meta(db, "kg_done", True)
    finally:
        source.close()


def question_inputs(db, qid):
    ps = [dict(p) for p in db.execute("""SELECT p.* FROM question_document d
        JOIN proposition p ON p.did=d.did WHERE d.qid=? ORDER BY d.rank,p.pos""", (qid,))]
    if not ps:
        return [], {}, []
    marks = ",".join("?" for _ in ps)
    ids = [p["idx"] for p in ps]
    entities, memberships = {}, defaultdict(set)
    for e in db.execute(f"""SELECT pe.idx,e.* FROM prop_entity pe JOIN entity e ON e.eid=pe.eid
            WHERE pe.idx IN ({marks})""", ids):
        entities[e["eid"]] = dict(e)
        memberships[e["idx"]].add(e["eid"])
    for p in ps:
        p["entities"] = memberships[p["idx"]]
    triples = [dict(t) for t in db.execute(f"SELECT * FROM triple WHERE idx IN ({marks})", ids)]
    return ps, entities, triples


def eligible_entity(e):
    # Do not turn mention-local/lexical clusters into asserted cross-document identity.
    return (e["level"] == "canonical" and bool(e["canonical"])
            and " ".join(e["name"].casefold().split()) not in GENERIC)


def select_pairs(ps, entities, triples, cosine, settings, document_count, qid):
    """Sparse undirected proposals. A single endpoint proposal is sufficient."""
    if len(cosine) != len(ps):
        raise ValueError("Cosine matrix/proposition alignment mismatch")
    usable = {eid for eid, e in entities.items() if eligible_entity(e)}
    es = [set(p["entities"]) & usable for p in ps]
    idf = {e: math.log((1+document_count)/(1+entities[e]["doc_frequency"])) + 1 for e in usable}
    rare = {e for e in usable if entities[e]["doc_frequency"] / max(1, document_count) <= settings["max_common_fraction"]}
    links = defaultdict(list)
    # Witness triples must be from THIS Top-8, never from the global external corpus.
    valid_indices = {p["idx"] for p in ps}
    for t in triples:
        if t["idx"] in valid_indices and t["s"] in usable and t["o"] in usable and t["s"] != t["o"]:
            links[tuple(sorted((t["s"], t["o"])))].append(t)
    candidates = {c: defaultdict(list) for c in CHANNELS}
    features = {}
    floor = settings["min_cosine"]
    for i, j in itertools.combinations(range(len(ps)), 2):
        a, b = ps[i], ps[j]
        sim = float(cosine[i][j])
        if not math.isfinite(sim):
            raise ValueError("Nonfinite cosine similarity")
        shared = es[i] & es[j]
        shared_ok = bool(shared & rare) or (bool(shared) and sim >= floor)
        witnesses = {}
        for x, y in itertools.product(es[i], es[j]):
            if x == y:
                continue
            for t in links.get(tuple(sorted((x, y))), ()):
                # A ubiquitous hub alone must not trigger a bridge proposal.
                if (x in rare and y in rare) or sim >= floor:
                    witnesses[t["tid"]] = t
        adjacent = a["did"] == b["did"] and abs(a["pos"]-b["pos"]) == 1
        features[i, j] = dict(cosine=sim, shared_entity_ids=sorted(shared),
                              bridge_triples=[witnesses[k] for k in sorted(witnesses)],
                              same_document=a["did"] == b["did"], adjacent=adjacent)
        scores = {}
        if shared_ok:
            scores["shared_entity"] = sum(idf[e] for e in shared)
        if witnesses:
            scores["direct_kg"] = max(min(idf[t["s"]], idf[t["o"]]) for t in witnesses.values())
        if sim >= floor:
            scores["semantic"] = sim
        if adjacent:
            scores["document_adjacent"] = 1.0
        for channel, score in scores.items():
            candidates[channel][i].append((score, sim, j))
            candidates[channel][j].append((score, sim, i))
    selected = defaultdict(lambda: defaultdict(list))
    for channel in CHANNELS:
        k = settings[channel + "_k"]
        for i, choices in candidates[channel].items():
            ranked = sorted(choices, key=lambda x: (-x[0], -x[1], ps[x[2]]["pid"]))
            picked = ranked[:k]
            # Reserve one of the SEMANTIC slots for an eligible cross-document pair.
            if k and channel == "semantic" and settings["reserve_cross_document"]:
                cross = [x for x in ranked if ps[i]["did"] != ps[x[2]]["did"]]
                if cross and not any(ps[i]["did"] != ps[x[2]]["did"] for x in picked):
                    picked[-1] = cross[0]
            for _, _, j in picked:
                selected[tuple(sorted((i, j)))][channel].append(ps[i]["pid"])
    records = []
    for (i, j), reasons in sorted(selected.items()):
        p, q = sorted((ps[i], ps[j]), key=lambda x: x["pid"])
        f = features[i, j]
        records.append(dict(pair_id=digest([p["pid"], q["pid"]]),
                            source=p["pid"], target=q["pid"],
                            indices=sorted((p["idx"], q["idx"])),
                            reasons={c: sorted(v) for c, v in reasons.items()}, **f))
    # Rejected samples are unassessed, NEVER negative training labels.
    rng = random.Random(digest([settings["seed"], qid, "rejected-audit"]))
    audit, audit_strata = [], {}
    for same in (True, False):
        pool = [ij for ij, f in features.items() if ij not in selected and f["same_document"] == same]
        take = min(settings["audit_per_stratum"], len(pool))
        name = "same_document" if same else "cross_document"
        audit_strata[name] = {"population": len(pool), "sample_size": take}
        for i, j in rng.sample(pool, take):
            audit.append(dict(source=ps[i]["pid"], target=ps[j]["pid"], stratum=name,
                              selection_probability=take/len(pool), status="unassessed", **features[i, j]))
    degrees = Counter()
    for i, j in selected:
        degrees[i] += 1
        degrees[j] += 1
    stats = dict(propositions=len(ps), possible_pairs=len(features), selected_pairs=len(records),
                 isolated_propositions=sum(i not in degrees for i in range(len(ps))),
                 cross_document_pairs=sum(not r["same_document"] for r in records),
                 channels={c: sum(c in r["reasons"] for r in records) for c in CHANNELS},
                 canonical_entity_propositions=sum(bool(x) for x in es))
    return dict(sample_id=qid, pairs=records, rejected_audit=audit, audit_strata=audit_strata, stats=stats)


def store_question(db, result):
    # A committed question and all its global pairs form one atomic unit.
    with db:
        for r in result["pairs"]:
            db.execute("INSERT OR IGNORE INTO pair VALUES (?,?)", r["indices"])
        db.execute("INSERT INTO question_result VALUES (?,?)", (result["sample_id"], json.dumps(result, ensure_ascii=False)))

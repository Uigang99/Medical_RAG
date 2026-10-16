"""Official-Train cohort and provenance preparation; no generation or QA fitting.

Frozen Test extractors are reused by scripts/run_rag2_train_kg_stage.py.
Question labels never enter document/proposition/frame files.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from medrag.training.graph_input_data import duplicate_groups, normalize_question

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'official_train10000_top8_kg_v1'
SEED = 20261016
CACHE = ROOT / 'datasets/filtering/rag2/llama3_8b_paper_compatible_three_anchor_v1/candidates/source_balanced32_rerank8_v1'
BENCH = ROOT / 'datasets/benchmark/mcq/unified'
SOURCES = ('cpg', 'pmc', 'pubmed', 'textbooks')
COUNTS = {'medmcqa': 7000, 'medqa': 3000}


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    tmp.replace(path)


def write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    tmp.replace(path)


def freeze(root, contract):
    value = {'contract': contract, 'contract_sha256': digest(contract)}
    p = root / 'manifest.json'
    if p.exists() and load(p) != value:
        raise RuntimeError(f'Immutable contract differs: {p}; use a new output root')
    if not p.exists():
        write_json(p, value)
    return value['contract_sha256']


def cache_manifest(dataset):
    path = CACHE / dataset / 'train' / 'candidate_manifest.json'
    return path


def verify_retrieval(m):
    if (m['retrieval_query_mode'] != 'no_rag_rationale_with_answer_conclusion'
        or m['rerank_query_mode'] != 'original_mcq_question_with_all_options'
        or m['query_prompt_version'] != 'rag2_paper_compatible_three_anchor_prompt_v1'
        or m['query_generation_policy_version'] != 'rag2_three_anchor_rationale_then_constrained_choice_v1'
        or m['candidate_pool_policy']['source_top_k'] != dict.fromkeys(SOURCES, 8)
        or m['candidate_pool_policy']['pool_size'] != 32
        or Path(m['query_embedding']['path']).name != 'MedCPT-Query-Encoder'
        or Path(m['reranker']['path']).name != 'MedCPT-Cross-Encoder'):
        raise RuntimeError('Retrieval cache does not match frozen Test retrieval semantics')


def select_cohort(root, event=lambda *a, **k: None):
    """Seeded subject-proportional selection; exclude duplicate stems, not hard QA.

    A deterministic 2x proposal pool bounds near-duplicate auditing. Duplicate
    detection uses the existing exact/rare-trigram >=.90 heuristic, not labels.
    Entire groups matching Test are excluded; one representative per other
    group prevents Train/Val duplicates. No replacements based on KG coverage.
    """
    test = list(rows(BENCH / 'all/test.jsonl'))
    if len(test) != 6545:
        raise RuntimeError('Final Test identity cohort must contain 6545 questions')
    pools = {}
    for ds, n in COUNTS.items():
        data = list(rows(BENCH / ds / 'train.jsonl'))
        if any(r['split'] != 'train' or r['dataset'] != ds for r in data):
            raise RuntimeError('Official Train source split mismatch')
        data.sort(key=lambda r: digest([SEED, r['id']]))
        pools[ds] = data[:2*n]
    proposed = [dict(r, _final_test=True) for r in test]
    proposed += [dict(r, _final_test=False) for pool in pools.values() for r in pool]
    groups, report = duplicate_groups(proposed)
    eligible = defaultdict(list)
    excluded = []
    for group in groups:
        train = [r for r in group if not r['_final_test']]
        if not train:
            continue
        if any(r['_final_test'] for r in group):
            excluded.extend({'id': r['id'], 'reason': 'test_stem_overlap'} for r in train)
            continue
        train.sort(key=lambda r: digest([SEED, r['id']]))
        eligible[train[0]['dataset']].append(train[0])
        excluded.extend({'id': r['id'], 'reason': 'duplicate_train_stem'} for r in train[1:])
    selected = []
    for ds, n in COUNTS.items():
        buckets = defaultdict(list)
        for r in eligible[ds]:
            buckets[str(r.get('subject') or 'unknown')].append(r)
        population = sum(map(len, buckets.values()))
        if population < n:
            raise RuntimeError(f'Insufficient decontaminated proposal pool for {ds}; do not silently change cohort policy')
        quotas = {k: n*len(v)//population for k, v in buckets.items()}
        order = sorted(buckets, key=lambda k: (-(n*len(buckets[k]) % population), k))
        for k in order[:n-sum(quotas.values())]:
            quotas[k] += 1
        chosen = []
        for k, v in sorted(buckets.items()):
            chosen += sorted(v, key=lambda r: digest([SEED, r['id']]))[:quotas[k]]
        # Exact per-dataset validation quotas; labels are not used.
        chosen.sort(key=lambda r: digest([SEED, 'split', r['id']]))
        for i, r in enumerate(chosen):
            selected.append({'sample_id': r['id'], 'dataset': ds,
                'official_split': 'train', 'split': 'validation' if i < n//10 else 'train',
                'subject': r.get('subject'), 'question': r['question'], 'options': r['options'],
                'answer': r['answer'], 'answers': r.get('answers', [r['answer']])})
    selected.sort(key=lambda r: (r['dataset'], r['sample_id']))
    inputs = {str(p): sha_file(p) for p in [BENCH/'all/test.jsonl', *(BENCH/d/'train.jsonl' for d in COUNTS)]}
    contract = {'version': VERSION, 'seed': SEED, 'counts': COUNTS, 'inputs': inputs,
        'split_counts': {'train': 9000, 'validation': 1000},
        'selection': 'seeded 2x proposal pool; subject-proportional; one per duplicate stem group; no QA correctness/coverage selection',
        'duplicate_policy': 'existing exact normalization + rare-trigram Jaccard>=0.90; not a semantic decontamination guarantee',
        'question_ids_sha256': digest([r['sample_id'] for r in selected])}
    ch = freeze(root/'cohort', contract)
    write_rows(root/'cohort/questions.jsonl', [{k:v for k,v in r.items() if k not in ('answer','answers')} for r in selected])
    write_rows(root/'cohort/labels.jsonl', [{'sample_id':r['sample_id'], 'answer':r['answer'], 'answers':r['answers']} for r in selected])
    write_rows(root/'cohort/excluded_duplicates.jsonl', excluded)
    write_json(root/'cohort/summary.json', {'questions':len(selected), 'duplicates':report,
        'counts':dict(Counter(f"{r['dataset']}/{r['split']}" for r in selected)),
        'subject_counts':dict(Counter(f"{r['dataset']}/{r['subject']}" for r in selected))})
    write_json(root/'cohort/COMPLETE.json', {'contract_sha256':ch, 'questions':len(selected)})


def prepare_documents(root, event=lambda *a, **k: None):
    import restore_rag2_retrieved_sentence_boundaries as restoration
    questions = {r['sample_id']:r for r in rows(root/'cohort/questions.jsonl')}
    stores = {s:restoration.Store(s) for s in SOURCES}
    documents, mapping, seen = {}, [], set()
    manifests = {}
    for ds in COUNTS:
        mp = cache_manifest(ds)
        verify_retrieval(load(mp))
        manifests[ds] = {'path':str(mp), 'sha256':sha_file(mp)}
        for r in rows(CACHE/ds/'train/candidates_top8.jsonl'):
            sid = r['sample_id']
            if sid not in questions:
                continue
            if sid in seen or r['question'] != questions[sid]['question'] or r['options'] != questions[sid]['options']:
                raise RuntimeError(f'Cached retrieval question mismatch: {sid}')
            seen.add(sid)
            docs = r['candidate_documents']
            if len(docs)!=8 or len({d['stable_id'] for d in docs})!=8 or r['initial_source_counts']!=dict.fromkeys(SOURCES,8):
                raise RuntimeError(f'Invalid balanced Top8 row: {sid}')
            for rank, d in enumerate(docs, 1):
                record = stores[d['source']].get(d['local_id'])
                if record['corpus_id'] != d['stable_id'] or record['doc_id'] != d['doc_id']:
                    raise RuntimeError(f'Corpus cache identity changed: {d["stable_id"]}')
                item = {k:d[k] for k in ('stable_id','source','doc_id','local_id')}
                item.update(text=record['text'], title=record.get('title',''), top_k_membership=[8])
                if d['stable_id'] in documents and documents[d['stable_id']] != item:
                    raise RuntimeError('Conflicting cached document payload')
                documents[d['stable_id']] = item
                mapping.append({'sample_id':sid, 'split':questions[sid]['split'], 'top_k':8,
                                'rank':rank, 'stable_id':d['stable_id']})
            event(len(seen), len(questions), 'question')
    missing = sorted(set(questions)-seen)
    if missing:
        write_json(root/'retrieval/missing_questions.json', missing)
        raise RuntimeError(f'{len(missing)} selected questions lack retrieval; IDs saved. No substitution or new retrieval was performed.')
    contract = {'version':VERSION, 'cohort_sha256':sha_file(root/'cohort/manifest.json'),
                'cache_manifests':manifests, 'questions':len(questions), 'top_k':8,
                'source_text_sha256':digest(sorted(documents.values(),key=lambda x:x['stable_id'])),
                'labels_or_query_rationales_in_document_generation':False}
    ch=freeze(root/'retrieval',contract)
    write_rows(root/'retrieval/documents.jsonl',sorted(documents.values(),key=lambda x:x['stable_id']))
    write_rows(root/'retrieval/question_document_membership.jsonl',sorted(mapping,key=lambda x:(x['sample_id'],x['rank'])))
    write_json(root/'retrieval/COMPLETE.json', {'contract_sha256':ch,'documents':len(documents),'questions':len(questions)})


def restore_documents(root, event=lambda *a, **k: None):
    import restore_rag2_retrieved_sentence_boundaries as method
    from concurrent.futures import ProcessPoolExecutor
    source=root/'retrieval/documents.jsonl'
    n=load(root/'retrieval/COMPLETE.json')['documents']
    ch=freeze(root/'restored',{'version':VERSION,'method':method.VERSION,'source_sha256':sha_file(source),
        'method_sha256':sha_file(method.__file__),'max_neighbors':8,'record_timeout_seconds':5})
    # Per-document atomic shards retain restoration work on interruption.
    shard=root/'restored/shards'; shard.mkdir(parents=True,exist_ok=True)
    docs=list(rows(source))
    pending=[]
    for d in docs:
        p=shard/(digest(d['stable_id'])+'.json')
        if p.exists():
            v=load(p)
            if v['contract_sha256']!=ch or v['row']['stable_id']!=d['stable_id']:
                raise RuntimeError('Restoration resume identity mismatch')
        else: pending.append(d)
    done=n-len(pending); event(done,n,'document')
    with ProcessPoolExecutor(max_workers=4, initializer=method.init_worker,initargs=(8,)) as pool:
        for result in pool.map(method.work,pending,chunksize=16):
            write_json(shard/(digest(result['stable_id'])+'.json'),{'contract_sha256':ch,'row':result})
            done+=1; event(done,n,'document')
    path=root/'restored/restored_chunks.jsonl'
    write_rows(path,(load(shard/(digest(d['stable_id'])+'.json'))['row'] for d in docs))
    write_json(root/'restored/source_scope.json',{'documents':n,'by_source':dict(Counter(d['source'] for d in docs)),
        'input_sha256':sha_file(path),'cohort_contract_sha256':load(root/'cohort/manifest.json')['contract_sha256']})
    write_json(root/'restored/COMPLETE.json',{'contract_sha256':ch,'documents':n})


def select_usable(root, event=lambda *a, **k: None):
    import prepare_rag2_gemma_union_usable_propositions as method
    repairs=method.load_repair(root/'repair/machine_valid_candidates.jsonl')
    contract={'version':VERSION,'original_sha256':sha_file(root/'propositions/document_propositions.jsonl'),
              'repair_sha256':sha_file(root/'repair/machine_valid_candidates.jsonl'),
              'method_sha256':sha_file(method.__file__),'selection_policy':'unchanged frozen union usable v2'}
    ch=freeze(root/'usable',contract)
    props=[]; audit=[]; counts=Counter(); docs=set()
    for original in rows(root/'propositions/document_propositions.jsonl'):
        sid=original['stable_id']; reasons=[]
        if original['source_status']!='boundary_expanded_candidate' or original.get('source_quality_flags'):
            reasons.append('source_quality_warning')
        replacement=repairs.get(sid)
        if replacement:
            origin='repair'; selected=replacement['parsed']['propositions']; text=replacement['source_text']
        elif method.clean_original(original):
            origin='original'; selected=original['propositions']; text=original['source_text']
        else:
            origin='none'; selected=[]; text=original['source_text']; reasons.append('unresolved_generation')
        rejected=Counter(); kept=0
        for i,p in enumerate(selected):
            reason=method.proposition_exclusion(p,text)
            if reason or reasons:
                rejected[reason or '+'.join(reasons)]+=1; continue
            claim,evidence=p['claim'].strip(),p['evidence'].strip()
            props.append(dict(proposition_id=method.proposition_id(sid,origin,i,claim,evidence),
                document_id=sid,stable_id=sid,source=original['source'],segment='top8',origin=origin,
                claim=claim,evidence=evidence,source_text_sha256=original['text_sha256']))
            counts[original['source']]+=1; docs.add(sid); kept+=1
        audit.append({'document_id':sid,'usable_propositions':kept,'document_exclusions':reasons,'proposition_exclusions':dict(rejected)})
    if not props: raise RuntimeError('No usable propositions; stop before KG generation')
    write_rows(root/'usable/usable_propositions.jsonl',props)
    write_rows(root/'usable/document_audit.jsonl',audit)
    write_json(root/'usable/summary.json',{'documents_audited':len(audit),'documents_with_usable_propositions':len(docs),
        'propositions':len(props),'by_source':dict(counts)})
    write_json(root/'usable/COMPLETE.json',{'contract_sha256':ch,'propositions':len(props)})


def export_question_views(root, event=lambda *a, **k: None):
    """IDs only: local Top8 evidence, never global neighbours or graph metrics."""
    import sqlite3
    db=root/'kg/graph.sqlite3'
    if not db.exists():
        raise FileNotFoundError(db)
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    mapping=defaultdict(list)
    for r in rows(root/'retrieval/question_document_membership.jsonl'):
        mapping[r['sample_id']].append(r)
    props=defaultdict(list)
    for r in rows(root/'usable/usable_propositions.jsonl'):
        props[r['document_id']].append(r['proposition_id'])
    edges_by_doc=defaultdict(list)
    for edge_id,prop_id,doc_id in c.execute('SELECT edge_id,proposition_id,document_id FROM binary_relation'):
        edges_by_doc[doc_id].append((edge_id,prop_id))
    output=[]
    for q in rows(root/'cohort/questions.jsonl'):
        docs=[r['stable_id'] for r in sorted(mapping[q['sample_id']],key=lambda r:r['rank'])]
        edges=[edge for d in docs for edge in edges_by_doc[d]]
        output.append(dict(q,document_ids=docs,usable_proposition_ids=[p for d in docs for p in props[d]],
            paired_proposition_ids=sorted({r[1] for r in edges}),binary_edge_ids=[r[0] for r in edges],
            status='available' if edges else 'no_binary_edge'))
        event(len(output),10000,'question')
    c.close()
    write_rows(root/'question_views.jsonl',output)
    write_json(root/'COMPLETE.json',{'version':VERSION,'questions':len(output),
        'empty_graph_questions':sum(not r['binary_edge_ids'] for r in output),
        'split_counts':dict(Counter(r['split'] for r in output)),
        'labels_path':'cohort/labels.jsonl','kg_path':'kg/graph.sqlite3',
        'question_views_sha256':sha_file(root/'question_views.jsonl')})

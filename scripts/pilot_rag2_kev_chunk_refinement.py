#!/usr/bin/env python3
"""Paired frozen-v1 versus explicit-v2 KEV study on new official-Train chunks."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag import kev_chunk_relations as schema
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, identity, ensure_contract
from medrag.training.graph_input_progress import Progress
from pilot_rag2_kev_chunk_relations import preflight


def read_rows(path):
    with path.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def prepare(a):
    paths={name:a.train_root/name for name in ('cohort/questions.jsonl', 'cohort/manifest.json',
        'retrieval/documents.jsonl','retrieval/question_document_membership.jsonl','retrieval/manifest.json')}
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError(path)
    previous=json.loads(a.exclude_cohort.read_text())
    excluded={digest(d['text']) for q in previous['questions'] for d in q['chunks']}
    docs={r['stable_id']:r for r in read_rows(paths['retrieval/documents.jsonl'])}
    mapping={}
    for r in read_rows(paths['retrieval/question_document_membership.jsonl']):
        mapping.setdefault(r['sample_id'],[]).append(r)
    candidates=read_rows(paths['cohort/questions.jsonl'])
    selected=[]; used=set(excluded); exclusions=Counter()
    for split in ('development','confirmation'):
        for dataset in ('medmcqa','medqa'):
            pool=sorted((r for r in candidates if r['dataset']==dataset and r['official_split']=='train'
                         and r['split']=='train'),key=lambda r:digest([a.seed,split,r['sample_id']]))
            n=0
            for r in pool:
                entries=sorted(mapping.get(r['sample_id'],[]),key=lambda x:x['rank'])
                if len(entries)!=8 or [x['rank'] for x in entries]!=list(range(1,9)):
                    raise ValueError('Incomplete fixed Top8: '+r['sample_id'])
                originals=[docs[x['stable_id']] for x in entries]
                hashes={digest(x['text']) for x in originals}
                if used & hashes:
                    exclusions['shared_exact_chunk']+=1; continue
                used.update(hashes)
                chunks=[{'id':x['stable_id'],'source_id':x['doc_id'],'source':x['source'],
                         'text':x['text'],'rank':i+1} for i,x in enumerate(originals)]
                selected.append({'case_id':f"T{len(selected)+1:02d}", 'sample_id':r['sample_id'],
                    'dataset':dataset,'official_split':'train','audit_split':split,'subject':r.get('subject'),
                    'chunks':chunks,'pairs':[]})
                n+=1
                if n==3: break
            if n!=3: raise ValueError('Insufficient disjoint official Train chunks')
    pairs=[]
    for q in selected:
        for i,j in combinations(range(8),2):
            x,y=sorted((q['chunks'][i],q['chunks'][j]),key=lambda d:(d['id'],d['text']))
            pid=digest([(d['id'],d['text']) for d in (x,y)])
            pairs.append({'pair_id':pid,'case_id':f"{q['case_id']}_{i+1}_{j+1}", 'A':x,'B':y,
                          'occurrences':[{'question':q['case_id'],'ranks':[i+1,j+1]}]})
            q['pairs'].append(pid)
        q['blind_review_pairs']=sorted(q['pairs'],key=lambda p:digest([a.seed,q['case_id'],p,'review']))[:4]
    if len({p['pair_id'] for p in pairs})!=len(pairs): raise ValueError('Repeated pair in disjoint draw')
    cohort={'version':'kev_chunk_train_refinement_v1','seed':a.seed,'questions':selected,'pairs':pairs}
    jobs=[]
    for version in ('v1','v2'):
        jobs.extend(dict(schema.make_job(p,t,version),prompt_version=version) for p in pairs for t in schema.TASKS)
    old=a.output/'manifest.json'
    commit=(json.loads(old.read_text())['contract']['git_commit_at_prepare'] if old.exists() else
            subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    model_paths=[a.adapter/'adapter_config.json',a.adapter/'adapter_model.safetensors',a.adapter/'head.pt',a.base/'config.json',*sorted(a.base.glob('*.safetensors'))]
    if len(model_paths)<5: raise FileNotFoundError('Base weight shards missing')
    contract={'version':cohort['version'],'hypothesis':'Explicit local-relation criteria improve paired labeling without trading all errors for abstentions',
        'seed':a.seed,'cohort_hash':digest(cohort),'jobs_hash':digest(jobs),'input_files':{k:identity(v,full_hash=True) for k,v in paths.items()},
        'exclude_previous':identity(a.exclude_cohort,full_hash=True),'excluded_counts':dict(exclusions),
        'split_counts':{'development':6,'confirmation':6},'datasets':{'medmcqa':6,'medqa':6},
        'scope':'Official Train only; question/answer not sent; exact cached Top8; no QA generation or retrieval changes; no tuning on confirmation',
        'models':[identity(p) for p in model_paths],'dtype':'bfloat16','gpu':1,'batch_rows':a.batch_rows,
        'code':{str(p.relative_to(ROOT)):identity(p,full_hash=True) for p in [Path(__file__),ROOT/'medrag/kev_chunk_relations.py',ROOT/'medrag/evidence_graph_runtime.py',ROOT/'scripts/pilot_rag2_kev_chunk_relations.py']},
        'git_commit_at_prepare':commit,'selection':'Same 48 pre-model blind-reference pairs; all 336 pairs predicted for coverage; versions frozen before execution',
        'quality_gate':{'clear_positive_precision':.85,'reference_positive_recall':.70,'minimum_positive_audit_per_relation':20,
                        'rule':'Improvement must not be abstention alone; inspect NO and UNCERTAIN and confirmation; undersupported labels inconclusive'},
        'stop':'One frozen v2 comparison in this run. No full-scale labeling; no post-hoc confidence threshold selection.'}
    h=ensure_contract(old,contract)
    atomic_json(a.output/'cohort.json',cohort); atomic_jsonl(a.output/'jobs.jsonl',jobs)
    wanted={pid for q in selected for pid in q['blind_review_pairs']}
    atomic_jsonl(a.output/'blind_review_inputs.jsonl',[p for p in pairs if p['pair_id'] in wanted])
    print(json.dumps({'questions':len(selected),'pairs':len(pairs),'jobs':len(jobs),'references_required':len(wanted),
                      'subjects':[q['subject'] for q in selected],'output':str(a.output)},ensure_ascii=False),flush=True)
    return cohort,jobs,h


def run(a,cohort,jobs,h):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise RuntimeError('GPU1 only')
    refpath=a.output/'blind_references.jsonl'
    if not refpath.is_file(): raise RuntimeError('Freeze blinded references before predictions')
    refs=read_rows(refpath); wanted={pid for q in cohort['questions'] for pid in q['blind_review_pairs']}
    if len(refs)!=len(wanted) or {r['pair_id'] for r in refs}!=wanted: raise ValueError('Reference coverage mismatch')
    for r in refs:
        if set(r['labels'])!=set(schema.TASKS) or set(r['labels'].values())-set(schema.CRITERIA):raise ValueError('Bad reference labels')
    ensure_contract(a.output/'frozen_reference_manifest.json',{'experiment_hash':h,'reference_identity':identity(refpath,full_hash=True)})
    cache=DecisionCache(a.output/'decisions',h)
    p=Progress(('context and references','load KEV GPU1','paired v1/v2 TRAIN12','graphs and summary'))
    p.weights[3]=2+len(jobs)*.001
    start=time.monotonic()
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            cached=sum(cache.get(j) is not None for j in jobs)
            p.log(f'[plan] official TRAIN12/336 pairs -> frozen v1/v2 -> 4032 decisions; cached={cached}; GPU1 batch={a.batch_rows}; previous shared-H200 estimate 12-20min, recalibrate on fresh work')
            p.stage(0,1); limits=preflight(a,jobs); p.update(); p.complete(f'No truncation; {limits}; frozen reference pairs={len(refs)}')
            p.stage(1,1)
            judge=None
            if cached<len(jobs):
                from transformers.utils import logging
                logging.set_verbosity_error(); logging.disable_progress_bar()
                judge=LocalDecisions(a.source,a.adapter,a.base,cache,a.batch_rows)
            p.update(); p.complete('KEV ready' if judge else 'all cached; no model load')
            def emit(event,**kw):
                if event=='stage':p.stage(**kw)
                elif event=='update':p.update(kw['amount'])
                elif event=='complete':p.complete(kw['message'])
            scored=time.monotonic()
            if judge: decisions=judge.score(jobs,2,emit)
            else:
                decisions={j['id']:cache.get(j) for j in jobs};p.stage(2,len(jobs),initial=len(jobs));p.complete('cached')
            seconds=time.monotonic()-scored
            p.stage(3,1); distributions={}
            for version in ('v1','v2'):
                selected=[j for j in jobs if j['prompt_version']==version]
                predictions,graphs,counts=schema.materialize(cohort,selected,decisions)
                atomic_jsonl(a.output/version/'predictions.jsonl',predictions)
                atomic_jsonl(a.output/version/'graphs.jsonl',graphs);distributions[version]=counts
            summary={'questions':12,'pairs':336,'decisions':len(jobs),'missing':0,'truncated':0,'fresh_decisions':len(jobs)-cached,
                'decision_seconds':seconds,'elapsed_seconds':time.monotonic()-start,'distribution':distributions,
                'batch_rows':a.batch_rows,'batch_parity':judge.parity if judge else None,'oom_splits':judge.oom_splits if judge else None,
                'batch_disabled':judge.batch_disabled if judge else None,'QA_evaluated':False}
            atomic_json(a.output/'summary.json',summary);p.update();p.complete(str(a.output/'summary.json'))
        except Exception as e:
            done=sum(cache.get(j) is not None for j in jobs)
            p.log(f'[FAILED] stage={p.index+1}/4 completed={done}/{len(jobs)} remaining={len(jobs)-done}; {type(e).__name__}: {e}; durable={a.output}; identical command resumes')
            raise
        finally:p.close()
    print('[complete] Frozen paired classification complete; direct qualitative review follows.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('prepare','preflight','run'),required=True)
    ap.add_argument('--train-root',type=Path,default=ROOT/'datasets/kg/rag2_official_train10000_top8_h200_throughput_v2')
    ap.add_argument('--exclude-cohort',type=Path,default=ROOT/'results/rag2_kev_chunk_relations_test24_v1/cohort.json')
    ap.add_argument('--output',type=Path,default=ROOT/'results/rag2_kev_chunk_refinement_train12_v2')
    ap.add_argument('--seed',type=int,default=2026102701)
    ap.add_argument('--batch-rows',type=int,default=8)
    ap.add_argument('--source',type=Path,default=ROOT.parent/'tools/kev-runtime')
    ap.add_argument('--adapter',type=Path,default=ROOT.parent/'models/kev-9b')
    ap.add_argument('--base',type=Path,default=ROOT.parent/'models/Qwen3.5-9B-Base')
    a=ap.parse_args()
    if not 1<=a.batch_rows<=64:ap.error('batch rows must be 1..64')
    c,j,h=prepare(a)
    if a.stage=='preflight':print(json.dumps(preflight(a,j)))
    elif a.stage=='run':run(a,c,j,h)

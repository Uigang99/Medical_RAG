#!/usr/bin/env python3
"""Bounded prompt-only development and one fresh official-Train confirmation."""
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
from medrag.kev_chunk_relations import TASKS, materialize
from medrag.kev27_chunk_prompts import make_job
from medrag.evidence_graph_runtime import LocalDecisions, DecisionCache
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, identity, ensure_contract
from medrag.training.graph_input_progress import Progress
from pilot_rag2_kev_chunk_relations import preflight
from review_rag2_kev_chunk_refinement import reference_metrics

OLD = ROOT/'results/rag2_kev_chunk_refinement_train12_v2'
BASELINE = ROOT/'results/rag2_kev27_chunk_refinement_train12_v1'
SEED = 2026102801


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare(a):
    old = json.loads((OLD/'cohort.json').read_text())
    if not (a.root/'development/cohort.json').exists():
        atomic_json(a.root/'development/cohort.json', old)
        atomic_jsonl(a.root/'development/references.jsonl', rows(OLD/'blind_references.jsonl'))
    path = a.root/'confirmation/cohort.json'
    if path.exists():
        return
    train = ROOT/'datasets/kg/rag2_official_train10000_top8_h200_throughput_v2'
    test = json.loads((ROOT/'results/rag2_kev_chunk_relations_test24_v1/cohort.json').read_text())
    used = {digest(d['text']) for c in (old, test) for q in c['questions'] for d in q['chunks']}
    excluded_ids = {q['sample_id'] for q in old['questions']}
    docs = {r['stable_id']:r for r in rows(train/'retrieval/documents.jsonl')}
    mapping = {}
    for r in rows(train/'retrieval/question_document_membership.jsonl'):
        mapping.setdefault(r['sample_id'], []).append(r)
    candidates = rows(train/'cohort/questions.jsonl'); selected = []
    for dataset in ('medmcqa', 'medqa'):
        pool = sorted((r for r in candidates if r['dataset']==dataset and r['official_split']=='train'
                       and r['split']=='train' and r['sample_id'] not in excluded_ids),
                      key=lambda r:digest([SEED, r['sample_id']]))
        n = 0
        for r in pool:
            entries = sorted(mapping[r['sample_id']], key=lambda x:x['rank'])
            assert [e['rank'] for e in entries] == list(range(1,9))
            source = [docs[e['stable_id']] for e in entries]
            hashes = {digest(d['text']) for d in source}
            if used & hashes: continue
            used.update(hashes)
            chunks = [{'id':d['stable_id'],'source_id':d['doc_id'],'source':d['source'],
                       'text':d['text'],'rank':i+1} for i,d in enumerate(source)]
            selected.append({'case_id':f'H{len(selected)+1:02d}', 'sample_id':r['sample_id'],
                             'dataset':dataset,'audit_split':'confirmation','chunks':chunks,'pairs':[]})
            n += 1
            if n == 3: break
        assert n == 3
    pairs = []
    for q in selected:
        for i,j in combinations(range(8),2):
            x,y = sorted((q['chunks'][i],q['chunks'][j]), key=lambda d:(d['id'],d['text']))
            pid = digest([(d['id'],d['text']) for d in (x,y)])
            pairs.append({'pair_id':pid,'case_id':f"{q['case_id']}_{i+1}_{j+1}",'A':x,'B':y})
            q['pairs'].append(pid)
        q['blind_review_pairs'] = sorted(q['pairs'], key=lambda x:digest([SEED,'review',x]))[:4]
    cohort = {'seed':SEED,'questions':selected,'pairs':pairs}
    assert len(pairs)==len({p['pair_id'] for p in pairs})==168
    atomic_json(path,cohort)
    wanted = {pid for q in selected for pid in q['blind_review_pairs']}
    atomic_jsonl(a.root/'confirmation/blind_inputs.jsonl',[p for p in pairs if p['pair_id'] in wanted])
    ensure_contract(a.root/'plan.json', {
        'version':'kev27_prompt_refinement_v1','hypothesis':'Explicit local positive criteria reduce false NO without reintroducing broad-topic YES',
        'development':'Old official Train12, all now development; 336 pairs',
        'confirmation':'New official Train6, 168 pairs; no exact chunks shared with old Train12 or TEST24',
        'confirmation_hash':digest(cohort),'seed':SEED,'max_updates':2,
        'fixed':'KEV27 pinned full BF16, GPU1 batch8, six independent Y/N/U decisions, raw chunks, no question/gold, no truncation',
        'selection':'Development macro F1 with unchanged frozen provisional references AND direct inspection of false positives/negatives; choose before confirmation',
        'minimum_worthwhile':'Development macro F1 +0.03; confirmation no regression; qualitative YES precision around 0.85 and recall around 0.70, otherwise restricted use only',
        'stop':'At most v3/v4; one confirmation v2 vs selected, no further tuning from confirmation; no full labeling or QA',
        'reference_caveat':'Assistant judgments, not independent medical expert gold; question-clustered small pilot',
        'input_files':{str(p.relative_to(ROOT)):identity(p,full_hash=True) for p in [train/'cohort/questions.jsonl',train/'retrieval/documents.jsonl',train/'retrieval/question_document_membership.jsonl']}})
    print('Prepared development336 + confirmation168 pairs; 24 new blind review pairs.',flush=True)


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise RuntimeError('GPU1 only')
    phase = a.root/a.phase
    cohort = json.loads((phase/'cohort.json').read_text()); refs = rows(phase/'references.jsonl')
    if a.phase=='confirmation':
        choice = json.loads((a.root/'selection.json').read_text())
        if set(a.versions) != {'v2',choice['selected_version']}:
            raise ValueError('Confirmation must compare v2 with frozen development selection')
        wanted={p for q in cohort['questions'] for p in q['blind_review_pairs']}
        if {r['pair_id'] for r in refs}!=wanted or len(refs)!=24:raise ValueError('Blind reference coverage')
    for r in refs:
        assert set(r['labels'])==set(TASKS) and set(r['labels'].values())<={'YES','NO','UNCERTAIN'}
    jobs=[make_job(p,t,v) for v in a.versions for p in cohort['pairs'] for t in TASKS]
    a.output = phase/('_'.join(a.versions))
    manifest=a.output/'manifest.json'
    commit=json.loads(manifest.read_text())['contract']['commit'] if manifest.exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract={'cohort':digest(cohort),'jobs':digest(jobs),'references':identity(phase/'references.jsonl',full_hash=True),
              'model':identity(a.adapter/'download_manifest.json',full_hash=True),'batch_rows':a.batch_rows,'gpu':1,
              'plan':identity(a.root/'plan.json',full_hash=True),'commit':commit,'phase':a.phase,'versions':a.versions,
              'code':identity(Path(__file__),full_hash=True),'dtype':'bf16','decision':'argmax, no thresholds'}
    h=ensure_contract(manifest,contract)
    atomic_jsonl(a.output/'jobs.jsonl',jobs)
    cache=DecisionCache(a.output/'decisions',h)
    p=Progress(('contract / token limits','load KEV27 GPU1',f'{a.phase} decisions','paired summary'))
    p.weights[1]=10;p.weights[2]=len({j['id'] for j in jobs})/6.82;p.weights[3]=2
    start=time.monotonic()
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p.stage(0,1); limits=preflight(a,jobs); cached=sum(cache.get(j) is not None for j in jobs)
            p.update();p.complete(f'{len(cohort["pairs"])} pairs; {len(jobs)} jobs; cached={cached}; {limits}')
            p.stage(1,1)
            import torch
            from transformers.utils import logging
            logging.set_verbosity_error();logging.disable_progress_bar()
            judge=None
            if cached<len(jobs):
                if torch.cuda.mem_get_info()[0]<70*1024**3: raise RuntimeError('Need 70GiB free on GPU1')
                judge=LocalDecisions(a.source,a.adapter,a.adapter,cache,a.batch_rows)
            p.update();p.complete('Frozen KEV27; no question/answer supplied')
            def emit(event,**kw):
                if event=='stage':p.stage(**kw)
                elif event=='update':p.update(kw['amount'])
                elif event=='complete':p.complete(kw['message'])
            at=time.monotonic()
            if judge: decisions=judge.score(jobs,2,emit)
            else:
                p.stage(2,len(jobs),initial=len(jobs));decisions={j['id']:cache.get(j) for j in jobs};p.complete('cached')
            seconds=time.monotonic()-at
            p.stage(3,1);summary={}
            for version in a.versions:
                selected=[j for j in jobs if j['prompt_version']==version]
                predictions,graphs,counts=materialize(cohort,selected,decisions)
                atomic_jsonl(a.output/version/'predictions.jsonl',predictions)
                atomic_jsonl(a.output/version/'graphs.jsonl',graphs)
                pred={r['pair_id']:r for r in predictions}
                metrics={rel:reference_metrics(refs,pred,rel) for rel in dict.fromkeys(t[1] for t in TASKS.values())}
                summary[version]={'distribution':counts,'reference_metrics':metrics,'macro_f1':sum(m['f1'] for m in metrics.values())/4}
            result={'versions':summary,'pairs':len(cohort['pairs']),'jobs':len(jobs),'missing':0,'truncated':0,
                    'decision_seconds':seconds,'elapsed_seconds':time.monotonic()-start,'batch_parity':judge.parity if judge else None,
                    'oom_splits':judge.oom_splits if judge else None,'QA_evaluated':False}
            atomic_json(a.output/'summary.json',result);p.update();p.complete(str(a.output/'summary.json'))
        except Exception as e:
            done=sum(cache.get(j) is not None for j in jobs)
            p.log(f'[FAILED] stage={p.index+1}/4 completed={done}/{len(jobs)}; durable={a.output}; identical command resumes; {e}')
            raise
        finally:p.close()
    print('[complete] '+json.dumps({v:summary[v]['macro_f1'] for v in summary}),flush=True)


def freeze_references(a):
    phase=a.root/'confirmation'
    annotations=json.loads((phase/'reference_annotations.json').read_text())
    inputs=rows(phase/'blind_inputs.jsonl')
    assert set(annotations)=={r['case_id'] for r in inputs}
    labels={'Y':'YES','N':'NO','U':'UNCERTAIN'}
    refs=[]
    for row in inputs:
        values,note=annotations[row['case_id']]
        assert len(values)==len(TASKS)
        refs.append({'pair_id':row['pair_id'],'case_id':row['case_id'],
                     'labels':{task:labels[value] for task,value in zip(TASKS,values)},'note':note})
    ensure_contract(phase/'frozen_references.json',{'references':digest(refs),'annotations':identity(phase/'reference_annotations.json',full_hash=True),
                                                 'timing':'Frozen before confirmation predictions; no model labels inspected'})
    atomic_jsonl(phase/'references.jsonl',refs)
    print('Frozen 24 confirmation pairs / 144 task labels.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('prepare','freeze-references','run'),required=True)
    ap.add_argument('--phase',choices=('development','confirmation'),default='development')
    ap.add_argument('--versions',nargs='+',default=['v3'])
    ap.add_argument('--root',type=Path,default=ROOT/'results/rag2_kev27_prompt_updates_v1')
    ap.add_argument('--adapter',type=Path,default=ROOT.parent/'models/kev-27b-v1.0')
    ap.add_argument('--source',type=Path,default=ROOT.parent/'tools/kev-runtime')
    ap.add_argument('--batch-rows',type=int,default=8)
    a=ap.parse_args()
    if a.stage=='prepare':prepare(a)
    elif a.stage=='freeze-references':freeze_references(a)
    else:run(a)

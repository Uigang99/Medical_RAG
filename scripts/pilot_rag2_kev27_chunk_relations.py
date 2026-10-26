#!/usr/bin/env python3
"""Pinned KEV27 download and fixed-input comparison with the completed KEV9 pilot."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medrag.training.graph_input_data import atomic_json,atomic_jsonl,digest,identity,ensure_contract
from medrag.training.graph_input_progress import Progress
from medrag.evidence_graph_runtime import DecisionCache,LocalDecisions
from medrag.kev_chunk_relations import materialize,TASKS
from pilot_rag2_kev_chunk_relations import preflight
from review_rag2_kev_chunk_refinement import reference_metrics

REPO='jaredpalmer/kev-27b'
REVISION='af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf'

def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def download(a):
    from huggingface_hub import HfApi,snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    p=Progress(('pinned model metadata','download KEV27','verify SHA256'))
    try:
        p.stage(0,1)
        info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
        files={x.rfilename:{'size':x.size,'sha256':x.lfs.sha256 if x.lfs else None}
               for x in info.siblings if '/' not in x.rfilename and x.rfilename.endswith(('.json','.safetensors','.pt','.jinja','.md'))}
        total=sum(x['size'] for x in files.values())
        def done():
            partial=list((a.adapter/'.cache/huggingface/download').glob('**/*.incomplete'))
            complete=0
            for name,spec in files.items():
                path=a.adapter/name
                if path.is_file() and path.stat().st_size==spec['size']:
                    complete+=spec['size']
                elif spec['sha256']:
                    # An interrupted Xet transfer and an HTTP retry may coexist.
                    # Count at most one partial file, never their duplicated bytes.
                    complete+=min(spec['size'],max((x.stat().st_size for x in partial if spec['sha256'] in x.name),default=0))
            return complete
        initial=done()
        if shutil.disk_usage(a.adapter.parent).free < (total-initial)*1.15+5*1024**3:raise RuntimeError('Insufficient disk space')
        plan={'repo':REPO,'revision':REVISION,'files':files,'bytes':total}
        ensure_contract(a.adapter/'download_plan.json',plan)
        p.update();p.complete(f'{REPO}@{REVISION}; {total/1e9:.2f} GB; no GPU')
        p.stage(1,total,initial=initial,unit='byte');p.weights[2]=max(1,total/5e8)
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut=pool.submit(snapshot_download,REPO,revision=REVISION,local_dir=a.adapter,allow_patterns=list(files),max_workers=8)
            last=initial
            while True:
                try:fut.result(timeout=10);break
                except TimeoutError:
                    current=done();p.update(max(0,current-last));last=max(last,current)
            p.update(total-p.done)
        p.complete(f'Snapshot downloaded; {a.adapter}')
        p.stage(2,total,unit='byte')
        for name,spec in files.items():
            path=a.adapter/name
            if path.stat().st_size!=spec['size']:raise RuntimeError('Size mismatch: '+name)
            if spec['sha256']:
                h=hashlib.sha256()
                with path.open('rb') as f:
                    while block:=f.read(16*1024**2):h.update(block);p.update(len(block))
                if h.hexdigest()!=spec['sha256']:raise RuntimeError('SHA256 mismatch: '+name)
            else:p.update(spec['size'])
        atomic_json(a.adapter/'download_manifest.json',dict(plan,status='complete',verified='file size and all supplied LFS SHA256'))
        p.complete('All checkpoint files verified')
    finally:p.close()

def prepare(a):
    cohort=json.loads((a.baseline/'cohort.json').read_text());jobs=rows(a.baseline/'jobs.jsonl')
    refs=rows(a.baseline/'blind_references.jsonl')
    frozen=json.loads((a.baseline/'frozen_reference_manifest.json').read_text())['contract']['reference_identity']
    if identity(a.baseline/'blind_references.jsonl',full_hash=True)['sha256']!=frozen['sha256']:raise ValueError('Baseline references changed')
    if len(cohort['pairs'])!=336 or len(jobs)!=4032:raise ValueError('Unexpected baseline cohort')
    dm=json.loads((a.adapter/'download_manifest.json').read_text())
    if dm['repo']!=REPO or dm['revision']!=REVISION or dm['status']!='complete':raise ValueError('Unverified checkpoint')
    for name,spec in dm['files'].items():
        if (a.adapter/name).stat().st_size!=spec['size']:raise ValueError('Checkpoint file changed: '+name)
    manifest=a.output/'manifest.json'
    commit=(json.loads(manifest.read_text())['contract']['git_commit_at_prepare'] if manifest.exists() else
            subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    contract={'version':'kev27_chunk_train12_v1','hypothesis':'Released KEV27 improves relation fidelity over KEV9 at fixed inputs and prompts',
        'cohort_hash':digest(cohort),'jobs_hash':digest(jobs),'refs_hash':digest(refs),'model_repo':REPO,'revision':REVISION,
        'model_manifest':identity(a.adapter/'download_manifest.json',full_hash=True),'gpu':1,'dtype':'bfloat16','batch_rows':a.batch_rows,
        'baseline':str(a.baseline),'baseline_summary':identity(a.baseline/'summary.json',full_hash=True),
        'decision_policy':'Independent YES/NO/UNCERTAIN argmax; identical v1 and v2 bodies; no tuning or QA labels',
        'confound':'Model replacement, not size-only ablation: base and full-weight training differ from KEV9',
        'review':'Frozen pre-model references plus same 124 previously reviewed edges and new blinded positive audit',
        'primary':'v2 fixed prompt: provisional-reference macro F1 versus KEV9 and direct positive review; v1 is a prespecified sensitivity check, not best-of selection',
        'quality_gate':{'clear_positive_fraction':.85,'reference_positive_recall':.70,'minimum_positives_per_label_prompt':20},
        'stop':'One frozen run; no full TEST scale-up or prompt selection using these results',
        'code':{str(x.relative_to(ROOT)):identity(x,full_hash=True) for x in [Path(__file__),ROOT/'medrag/evidence_graph_runtime.py',ROOT/'medrag/kev_chunk_relations.py']},
        'git_commit_at_prepare':commit}
    h=ensure_contract(manifest,contract)
    atomic_json(a.output/'cohort.json',cohort);atomic_jsonl(a.output/'jobs.jsonl',jobs)
    atomic_jsonl(a.output/'blind_references.jsonl',refs)
    return cohort,jobs,refs,h

def review(a):
    cohort=json.loads((a.output/'cohort.json').read_text());pairs={x['pair_id']:x for x in cohort['pairs']}
    pred=rows(a.output/'v2/predictions.jsonl')
    old_reviews={r['audit_id']:r for r in rows(a.baseline/'positive_audit_reviews.jsonl')}
    splits={q['case_id']:q['audit_split'] for q in cohort['questions']}
    chosen=[]
    for relation in dict.fromkeys(t[1] for t in TASKS.values()):
        candidates=[{'audit_id':r['case_id']+':'+t,'pair_id':r['pair_id'],'case_id':r['case_id'],'task':t,'relation':relation}
                    for r in pred for t in TASKS if TASKS[t][1]==relation and r['answers'][t]['choice']=='YES']
        candidates.sort(key=lambda x:digest(['audit_v2',x['audit_id']]))
        selected=[]
        for split in ('development','confirmation'):
            selected.extend([x for x in candidates if splits[x['case_id'].split('_')[0]]==split][:10])
        used={x['audit_id'] for x in selected}
        selected.extend([x for x in candidates if x['audit_id'] not in used][:max(0,20-len(selected))])
        chosen.extend(selected)
    for x in chosen:
        pair=pairs[x['pair_id']];x.update(A=pair['A'],B=pair['B'])
    atomic_jsonl(a.output/'positive_audit_inputs.jsonl',chosen)
    reused=[old_reviews[x['audit_id']] for x in chosen if x['audit_id'] in old_reviews]
    atomic_jsonl(a.output/'positive_audit_reused.jsonl',reused)
    atomic_jsonl(a.output/'positive_audit_new_inputs.jsonl',[x for x in chosen if x['audit_id'] not in old_reviews])
    result={'primary_prompt':'v2','sampled':len(chosen),'reused_frozen_reviews':len(reused),'new_review_required':len(chosen)-len(reused)}
    path=a.output/'positive_audit_new_reviews.jsonl'
    if path.exists():
        new=rows(path);byid={x['audit_id']:x for x in reused+new}
        if len(byid)!=len(reused+new) or set(byid)!={x['audit_id'] for x in chosen}:raise ValueError('Incomplete/duplicate qualitative audit')
        from collections import Counter
        result['positive_audit']={}
        for rel in dict.fromkeys(t[1] for t in TASKS.values()):
            sample=[x for x in chosen if x['relation']==rel];counts=Counter(byid[x['audit_id']]['verdict'] for x in sample)
            n=len(sample);result['positive_audit'][rel]={'n':n,'counts':dict(counts),'clear_fraction':counts['CLEAR']/n if n else None}
        result['caveat']='Assistant source review; not independent expert gold; small topic-clustered sample; not QA accuracy'
    atomic_json(a.output/'qualitative_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('GPU1 only')
    cohort,jobs,refs,h=prepare(a);cache=DecisionCache(a.output/'decisions',h)
    p=Progress(('fixed inputs and context','load KEV27 GPU1','v1/v2 TRAIN12 decisions','paired summary'))
    p.weights[3]=6;start=time.monotonic()
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p.stage(0,1);lim=preflight(a,jobs);cached=sum(cache.get(j) is not None for j in jobs)
            p.update();p.complete(f'336 pairs, 4032 decisions, cached={cached}; {lim}; same prompts, no QA')
            import torch
            p.stage(1,1)
            judge=None
            if cached<len(jobs):
                free,_=torch.cuda.mem_get_info()
                if free<70*1024**3:raise RuntimeError(f'Need >=70 GiB free on GPU1; have {free/1024**3:.1f}; do not terminate other jobs')
                from transformers.utils import logging
                logging.set_verbosity_error();logging.disable_progress_bar()
                judge=LocalDecisions(a.source,a.adapter,a.adapter,cache,a.batch_rows)
            p.update();p.complete('Full BF16 KEV27; no base substitution; ETA calibrates on fresh batches')
            def emit(event,**kw):
                if event=='stage':p.stage(**kw)
                elif event=='update':p.update(kw['amount'])
                elif event=='complete':p.complete(kw['message'])
            at=time.monotonic()
            if judge:decisions=judge.score(jobs,2,emit)
            else:
                p.stage(2,len(jobs),initial=len(jobs));decisions={j['id']:cache.get(j) for j in jobs};p.complete('All cached')
            decision_seconds=time.monotonic()-at
            p.stage(3,1);distribution={};metrics={};changes={}
            for v in ('v1','v2'):
                selected=[j for j in jobs if j['prompt_version']==v]
                predictions,graphs,counts=materialize(cohort,selected,decisions)
                atomic_jsonl(a.output/v/'predictions.jsonl',predictions);atomic_jsonl(a.output/v/'graphs.jsonl',graphs)
                distribution[v]=counts;pred={x['pair_id']:x for x in predictions}
                old={x['pair_id']:x for x in rows(a.baseline/v/'predictions.jsonl')}
                metrics[v]={rel:{'kev9':reference_metrics(refs,old,rel),'kev27':reference_metrics(refs,pred,rel)} for rel in dict.fromkeys(x[1] for x in TASKS.values())}
                changes[v]=[{'case_id':r['case_id'],'pair_id':r['pair_id'],'task':t,'kev9':old[r['pair_id']]['answers'][t]['choice'],'kev27':r['answers'][t]['choice']} for r in predictions for t in TASKS if old[r['pair_id']]['answers'][t]['choice']!=r['answers'][t]['choice']]
            atomic_json(a.output/'changed_decisions.json',changes)
            summary={'questions':12,'pairs':336,'decisions':4032,'missing':0,'truncated':0,'fresh_decisions':len(jobs)-cached,
                'decision_seconds':decision_seconds,'elapsed_seconds':time.monotonic()-start,'distribution':distribution,'reference_comparison':metrics,
                'batch_rows':a.batch_rows,'batch_parity':judge.parity if judge else None,'oom_splits':judge.oom_splits if judge else None,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,
                'QA_evaluated':False,'reference_caveat':'Same assistant provisional reference, not independent expert gold'}
            atomic_json(a.output/'summary.json',summary);p.update();p.complete(str(a.output/'summary.json'))
        except Exception as e:
            done=sum(cache.get(j) is not None for j in jobs)
            p.log(f'[FAILED] stage={p.index+1}/4 completed={done}/4032 remaining={4032-done}; {e}; durable={a.output}; identical command resumes')
            raise
        finally:p.close()
    print('[complete] KEV27 fixed comparison finished; qualitative audit remains.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('download','preflight','run','review'),required=True)
    ap.add_argument('--baseline',type=Path,default=ROOT/'results/rag2_kev_chunk_refinement_train12_v2')
    ap.add_argument('--output',type=Path,default=ROOT/'results/rag2_kev27_chunk_refinement_train12_v1')
    ap.add_argument('--adapter',type=Path,default=ROOT.parent/'models/kev-27b-v1.0')
    ap.add_argument('--source',type=Path,default=ROOT.parent/'tools/kev-runtime')
    ap.add_argument('--batch-rows',type=int,default=8)
    a=ap.parse_args()
    if not 1<=a.batch_rows<=32:ap.error('batch rows must be 1..32')
    if a.stage=='download':download(a)
    elif a.stage=='review':review(a)
    elif a.stage=='preflight':c,j,r,h=prepare(a);print(preflight(a,j))
    else:run(a)

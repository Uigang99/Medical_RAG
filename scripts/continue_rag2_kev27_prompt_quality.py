#!/usr/bin/env python3
"""Further KEV27 prompt development and untouched official-Train confirmation."""
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
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, identity, ensure_contract
from medrag.training.graph_input_progress import Progress
from pilot_rag2_kev_chunk_relations import preflight
from review_rag2_kev_chunk_refinement import reference_metrics

PREVIOUS = ROOT/'results/rag2_kev27_prompt_updates_v1'
ORIGINAL = ROOT/'results/rag2_kev_chunk_refinement_train12_v2'
TEST24 = ROOT/'results/rag2_kev_chunk_relations_test24_v1/cohort.json'
TRAIN = ROOT/'datasets/kg/rag2_official_train10000_top8_h200_throughput_v2'
SEED = 2026102802


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def immutable_json(path, value):
    if path.exists():
        existing=json.loads(path.read_text())
        if existing==value:return
        # Repair only the wrapper written by the first prepare invocation.
        if existing.get('contract')!=value or set(existing)!={'contract','contract_sha256'}:
            raise ValueError(f'Existing cohort differs: {path}')
    atomic_json(path,value)


def prepare(a):
    development=a.root/'development'; confirmation=a.root/'confirmation'
    old=json.loads((ORIGINAL/'cohort.json').read_text())
    prior=json.loads((PREVIOUS/'confirmation/cohort.json').read_text())
    earlier=json.loads(TEST24.read_text())
    dev={'seed':'prior_official_train18','questions':old['questions']+prior['questions'],
         'pairs':old['pairs']+prior['pairs']}
    assert len(dev['questions'])==18 and len(dev['pairs'])==504
    development.mkdir(parents=True,exist_ok=True)
    immutable_json(development/'cohort.json',dev)
    devrefs=read_rows(ORIGINAL/'blind_references.jsonl')+read_rows(PREVIOUS/'confirmation/references.jsonl')
    assert len(devrefs)==72 and len({r['pair_id'] for r in devrefs})==72
    if (development/'references.jsonl').exists():
        assert read_rows(development/'references.jsonl')==devrefs
    else: atomic_jsonl(development/'references.jsonl',devrefs)
    confirm_path=confirmation/'cohort.json'
    used={digest(d['text']) for cohort in (dev,earlier) for q in cohort['questions'] for d in q['chunks']}
    excluded={q['sample_id'] for q in dev['questions']}
    docs={r['stable_id']:r for r in read_rows(TRAIN/'retrieval/documents.jsonl')}
    memberships={}
    for r in read_rows(TRAIN/'retrieval/question_document_membership.jsonl'):
        memberships.setdefault(r['sample_id'],[]).append(r)
    selected=[]; skipped=Counter()
    for dataset in ('medmcqa','medqa'):
        pool=sorted((r for r in read_rows(TRAIN/'cohort/questions.jsonl')
                     if r['dataset']==dataset and r['official_split']=='train'
                     and r['split']=='train' and r['sample_id'] not in excluded),
                    key=lambda r:digest([SEED,r['sample_id']]))
        found=0
        for r in pool:
            entries=sorted(memberships.get(r['sample_id'],[]),key=lambda x:x['rank'])
            if len(entries)!=8 or [e['rank'] for e in entries]!=list(range(1,9)):
                raise ValueError('Missing fixed Top8: '+r['sample_id'])
            source=[docs[e['stable_id']] for e in entries]
            hashes={digest(d['text']) for d in source}
            if used & hashes:
                skipped['shared_exact_chunk']+=1;continue
            used.update(hashes)
            chunks=[{'id':d['stable_id'],'source_id':d['doc_id'],'source':d['source'],
                     'text':d['text'],'rank':i+1} for i,d in enumerate(source)]
            selected.append({'case_id':f'R{len(selected)+1:02d}', 'sample_id':r['sample_id'],
                             'dataset':dataset,'official_split':'train','audit_split':'confirmation',
                             'chunks':chunks,'pairs':[]})
            found+=1
            if found==6:break
        assert found==6
    pairs=[]
    for q in selected:
        for i,j in combinations(range(8),2):
            x,y=sorted((q['chunks'][i],q['chunks'][j]),key=lambda d:(d['id'],d['text']))
            pid=digest([(d['id'],d['text']) for d in (x,y)])
            pairs.append({'pair_id':pid,'case_id':f"{q['case_id']}_{i+1}_{j+1}",'A':x,'B':y})
            q['pairs'].append(pid)
        q['blind_review_pairs']=sorted(q['pairs'],key=lambda pid:digest([SEED,'blind',pid]))[:4]
    assert len(selected)==12 and len(pairs)==336 and len({p['pair_id'] for p in pairs})==336
    cohort={'seed':SEED,'questions':selected,'pairs':pairs}
    wanted={pid for q in selected for pid in q['blind_review_pairs']}
    assert len(wanted)==48
    inputs=[p for p in pairs if p['pair_id'] in wanted]
    immutable_json(confirm_path,cohort)
    atomic_jsonl(confirmation/'blind_inputs.jsonl',inputs)
    ensure_contract(a.root/'plan.json',{
        'hypothesis':'A question-independent decision prompt can recover grounded relational edges without broad-topic false positives',
        'scope':'official Train only; no TEST tuning, retrieval, QA, or graph use yet',
        'development':'prior official Train18, 504 pairs, 72 already reviewed blind pairs',
        'confirmation':'new official Train12, 336 pairs, 48 blind pairs; no exact chunk shared with prior Train18 or TEST24',
        'cohort_hash':digest(cohort),'seed':SEED,'skipped':dict(skipped),
        'model':'full BF16 KEV27 pinned af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf, GPU1 batch8',
        'inputs':'exact 120-word retrieved chunk pairs, no question/options/gold',
        'labels':'six independent YES/NO/UNCERTAIN decisions; no score thresholds',
        'development_selection':'frozen-reference macro F1 + independent positive-edge audit; pick version before confirmation',
        'expansion_gate':'each graph label direct YES precision >=0.80 and recall >=0.65 on new confirmation, with sufficiently diverse positives; otherwise no full expansion',
        'uncertainty':'assistant references, not independent medical-expert gold; 48 pairs may be too small for a definitive gate',
        'stop':'at most two new prompt revisions after v4, one untouched confirmation, then stop or narrow scope',
        'source_files':{str(p.relative_to(ROOT)):identity(p,full_hash=True) for p in
                        [TRAIN/'cohort/questions.jsonl',TRAIN/'retrieval/documents.jsonl',
                         TRAIN/'retrieval/question_document_membership.jsonl']}})
    print(json.dumps({'development_questions':18,'confirmation_questions':12,'confirmation_pairs':336,
                      'blind_pairs':len(inputs),'skipped':dict(skipped),'root':str(a.root)}),flush=True)


def freeze(a):
    confirm=a.root/'confirmation'; inputs=read_rows(confirm/'blind_inputs.jsonl')
    annotations=json.loads((confirm/'reference_annotations.json').read_text())
    assert len(inputs)==48 and set(annotations)=={r['case_id'] for r in inputs}
    letters={'Y':'YES','N':'NO','U':'UNCERTAIN'};refs=[]
    for r in inputs:
        values,note=annotations[r['case_id']]
        assert len(values)==len(TASKS) and set(values)<=set(letters)
        refs.append({'case_id':r['case_id'],'pair_id':r['pair_id'],
                     'labels':{task:letters[val] for task,val in zip(TASKS,values)},'note':note})
    ensure_contract(confirm/'frozen_references.json',{
        'references_hash':digest(refs),'annotations':identity(confirm/'reference_annotations.json',full_hash=True),
        'timing':'assistant blind annotation before KEV confirmation scores'})
    if (confirm/'references.jsonl').exists():
        assert read_rows(confirm/'references.jsonl')==refs
    else: atomic_jsonl(confirm/'references.jsonl',refs)
    print(f'Frozen {len(refs)} blind pair references.',flush=True)


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('GPU1 only')
    phase=a.root/a.phase;cohort=json.loads((phase/'cohort.json').read_text());refs=read_rows(phase/'references.jsonl')
    if a.phase=='confirmation':
        selection=json.loads((a.root/'selection.json').read_text())
        if set(a.versions)!={'v2',selection['selected_version']}:
            raise ValueError('Confirmation must compare v2 with frozen development selection')
        wanted={pid for q in cohort['questions'] for pid in q['blind_review_pairs']}
        if len(refs)!=48 or {r['pair_id'] for r in refs}!=wanted:
            raise ValueError('Blind confirmation coverage')
        if not (phase/'frozen_references.json').exists():raise ValueError('References not frozen')
    for r in refs:
        assert set(r['labels'])==set(TASKS) and set(r['labels'].values())<={'YES','NO','UNCERTAIN'}
    jobs=[make_job(pair,task,version) for version in a.versions for pair in cohort['pairs'] for task in TASKS]
    output=phase/('_'.join(a.versions));output.mkdir(parents=True,exist_ok=True);a.output=output
    manifest=output/'manifest.json';commit=(json.loads(manifest.read_text())['contract']['commit'] if manifest.exists()
                                       else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    contract={'cohort':digest(cohort),'jobs':digest(jobs),'references':identity(phase/'references.jsonl',full_hash=True),
              'model':identity(a.adapter/'download_manifest.json',full_hash=True),'batch_rows':a.batch_rows,'gpu':1,
              'plan':identity(a.root/'plan.json',full_hash=True),'commit':commit,'phase':a.phase,'versions':a.versions,
              'code':identity(Path(__file__),full_hash=True),'dtype':'bf16','decision':'argmax; no thresholds'}
    h=ensure_contract(manifest,contract);atomic_jsonl(output/'jobs.jsonl',jobs)
    cache=DecisionCache(output/'decisions',h)
    p=Progress(('contract / token limits','load KEV27 GPU1',f'{a.phase} decisions','paired summary'))
    p.weights[1]=10;p.weights[2]=len({j['id'] for j in jobs})/6.82;p.weights[3]=2
    start=time.monotonic()
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p.stage(0,1);limits=preflight(a,jobs);cached=sum(cache.get(j) is not None for j in jobs)
            p.update();p.complete(f'{len(cohort["pairs"])} pairs; {len(jobs)} jobs; cached={cached}; {limits}')
            p.stage(1,1);judge=None
            if cached<len(jobs):
                import torch
                from transformers.utils import logging
                logging.set_verbosity_error();logging.disable_progress_bar()
                if torch.cuda.mem_get_info()[0]<70*1024**3:raise RuntimeError('Need 70GiB free on GPU1')
                judge=LocalDecisions(a.source,a.adapter,a.adapter,cache,a.batch_rows)
            p.update();p.complete('Frozen KEV27; no QA data supplied')
            def emit(event,**kw):
                if event=='stage':p.stage(**kw)
                elif event=='update':p.update(kw['amount'])
                elif event=='complete':p.complete(kw['message'])
            t=time.monotonic()
            if judge:decisions=judge.score(jobs,2,emit)
            else:
                p.stage(2,len(jobs),initial=len(jobs));decisions={j['id']:cache.get(j) for j in jobs};p.complete('cached')
            seconds=time.monotonic()-t;p.stage(3,1);summary={}
            for version in a.versions:
                subset=[j for j in jobs if j['prompt_version']==version]
                predictions,graphs,counts=materialize(cohort,subset,decisions)
                atomic_jsonl(output/version/'predictions.jsonl',predictions)
                atomic_jsonl(output/version/'graphs.jsonl',graphs)
                indexed={r['pair_id']:r for r in predictions}
                metrics={rel:reference_metrics(refs,indexed,rel) for rel in dict.fromkeys(t[1] for t in TASKS.values())}
                summary[version]={'distribution':counts,'reference_metrics':metrics,
                                  'macro_f1':sum(m['f1'] for m in metrics.values())/4}
            atomic_json(output/'summary.json',{'versions':summary,'pairs':len(cohort['pairs']),
                'reviewed_pairs':len(refs),'jobs':len(jobs),'missing':0,'truncated':0,
                'decision_seconds':seconds,'elapsed_seconds':time.monotonic()-start,
                'batch_parity':judge.parity if judge else None,'oom_splits':judge.oom_splits if judge else None,
                'QA_evaluated':False})
            p.update();p.complete(str(output/'summary.json'))
        except Exception as exc:
            completed=sum(cache.get(j) is not None for j in jobs)
            p.log(f'[FAILED] stage={p.index+1}/4 completed={completed}/{len(jobs)} remaining={len(jobs)-completed}; durable={output}; same command resumes; {exc}')
            raise
        finally:p.close()
    print('[complete] '+json.dumps({v:summary[v]['macro_f1'] for v in summary}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('prepare','freeze-references','run'),required=True)
    ap.add_argument('--phase',choices=('development','confirmation'),default='development')
    ap.add_argument('--versions',nargs='+',default=['v5'])
    ap.add_argument('--root',type=Path,default=ROOT/'results/rag2_kev27_prompt_quality_round2_v1')
    ap.add_argument('--adapter',type=Path,default=ROOT.parent/'models/kev-27b-v1.0')
    ap.add_argument('--source',type=Path,default=ROOT.parent/'tools/kev-runtime')
    ap.add_argument('--batch-rows',type=int,default=8)
    a=ap.parse_args()
    if a.stage=='prepare':prepare(a)
    elif a.stage=='freeze-references':freeze(a)
    else:run(a)

#!/usr/bin/env python3
"""Bounded primary/secondary KEV information-role pilot; no QA or bulk labeling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.evidence_graph_runtime import DecisionCache, LocalDecisions
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, jsonl
from medrag.training.graph_input_progress import Progress
from scripts.pilot_rag2_kev_information_roles import ROLES, cohort as old_cohort, wilson

DEFAULT = ROOT / 'results/rag2_kev_primary_roles_pilot_v1'
OLD = ROOT / 'results/rag2_kev_abc_train_input_pilot_v1/cohort.json'
DATA = ROOT / 'datasets/kg/rag2_official_train10000_top8_h200_throughput_v2'


def normalized(text):
    return ' '.join(text.casefold().split())


def prepare(root):
    paths = {'views': DATA/'question_views.jsonl', 'usable': DATA/'usable/usable_propositions.jsonl', 'old': OLD}
    contract = {'version': 'primary_roles_cohort_v1', 'seed': '20261025-main-role',
                'inputs': {k: identity(v) for k,v in paths.items()},
                'development': 'Old 92 development propositions; previous confirmation is excluded, not reused.',
                'confirmation': '32 hash-sampled source-document representatives per source; one proposition per document; all official training evidence; old documents/exact texts excluded.',
                'no_question_answer_in_model': True, 'max_prompt_versions': 4}
    root.mkdir(parents=True, exist_ok=True)
    h = ensure_contract(root/'sampling_manifest.json', contract)
    out = root/'cohort.json'
    if out.exists():
        if json.loads(out.read_text())['contract'] != h:
            raise RuntimeError('Cohort contract mismatch')
        print(f'Cached {out}')
        return
    progress = Progress(('sampling contract', 'index official train', 'sample unseen documents', 'save cohort'))
    try:
        progress.log('[plan] old development92 + new source-balanced confirmation128; no KEV, QA labels, or TEST inputs')
        progress.stage(0,1)
        old = old_cohort(OLD)
        raw_old = json.loads(OLD.read_text())['pairs']
        used_docs = {p[s]['document_id'] for p in raw_old for s in ('P1','P2')}
        used_texts = {normalized(r['text']) for r in old}
        dev = [r for r in old if r['split']=='development']
        progress.update(); progress.complete('Fixed seed and document exclusions before output inspection')
        progress.stage(1,10000,unit='question')
        eligible = set()
        for view in jsonl(paths['views']):
            if view['official_split']=='train' and view['split']=='train':
                eligible.update(view['usable_proposition_ids'])
            progress.update()
        progress.complete(f'eligible propositions={len(eligible)}')
        progress.stage(2, len(eligible), unit='proposition')
        by_doc = {}
        for row in jsonl(paths['usable']):
            if row['proposition_id'] not in eligible:
                continue
            progress.update()
            if row['document_id'] in used_docs or normalized(row['claim']) in used_texts:
                continue
            key = digest([contract['seed'],row['proposition_id']])
            did = row['document_id']
            if did not in by_doc or key < by_doc[did][0]:
                by_doc[did] = (key,row)
        groups = defaultdict(list)
        for did,(_,row) in by_doc.items():
            groups[row['source']].append((digest([contract['seed'],did]),row))
        confirmation = []
        for source in ('textbooks','cpg','pubmed','pmc'):
            taken = 0
            for _,row in sorted(groups[source]):
                key = normalized(row['claim'])
                if key in used_texts:
                    continue
                used_texts.add(key)
                confirmation.append({'id':f'H{len(confirmation)+1:03}', 'split':'confirmation',
                    'source':source,'document_id':row['document_id'], 'proposition_id':row['proposition_id'],
                    'text':row['claim']})
                taken += 1
                if taken == 32:
                    break
            if taken != 32:
                raise RuntimeError(f'Insufficient unseen {source} documents: {taken}')
        progress.complete('32 new documents/source, no old document or exact-text overlap')
        progress.stage(3,1)
        atomic_json(out, {'contract':h,'rows':dev+confirmation})
        progress.update();progress.complete(str(out))
    finally:
        progress.close()


def make_job(row, spec, primary=None):
    if primary is None:
        qs = {'primary':{'type':'choice','instructions':spec['primary_instruction'],
                         'criteria':spec['primary_criteria']}}
    else:
        qs = {r:{'type':'choice','instructions':spec['secondary_instruction']+'\n'+spec['secondary_roles'][r],
                 'criteria':spec['secondary_criteria']} for r in ROLES if r != primary}
    body = {'state':{'proposition':row['text']},'questions':qs}
    if primary is not None:
        body['state']['selected_primary_role'] = primary
    return {'id':digest(body),'body':body,'meta':{'id':row['id']}}


def evaluate(rows, refs):
    exact=acceptable=covered=abstain=0
    sub_valid=sub_total=sub_needed=sub_recovered=0
    confusion=Counter(); by_role=defaultdict(lambda:Counter(n=0,exact=0,acceptable=0))
    for row in rows:
        r=refs[row['id']]; p=row['primary']['choice']
        exact += p == r['primary']; acceptable += p in r['acceptable_primary']
        abstain += p in ('OTHER','UNCERTAIN')
        ss = {k for k,v in row['secondary'].items() if v['choice']=='YES'}
        covered += bool(({p}|ss)&set(r['acceptable_primary']))
        allowed=set(r['allowed_secondary'])|set(r['acceptable_primary'])
        needed=set(r.get('essential_secondary',[]))-{p}
        sub_valid+=len(ss&allowed); sub_total+=len(ss)
        sub_needed+=len(needed); sub_recovered+=len(needed&ss)
        confusion[(r['primary'],p)]+=1
        by_role[r['primary']].update(n=1,exact=int(p==r['primary']),acceptable=int(p in r['acceptable_primary']))
    return {'n':len(rows),'primary_exact_n':exact,'primary_acceptable_n':acceptable,
            'primary_exact_rate':exact/len(rows),'primary_acceptable_rate':acceptable/len(rows),
            'primary_acceptable_ci95':wilson(acceptable,len(rows)),'core_anywhere_n':covered,
            'unresolved_n':abstain,'secondary_valid_n':sub_valid,'secondary_predicted_n':sub_total,
            'secondary_precision':sub_valid/sub_total if sub_total else None,
            'essential_secondary_n':sub_needed,'essential_secondary_recovered':sub_recovered,
            'by_primary_role':dict(by_role),
            'confusion':[{'reference':a,'prediction':b,'n':n} for (a,b),n in sorted(confusion.items())],
            'interpretation':'Assistant pre-output priority reference agreement, not independently measured gold accuracy.'}


def run(a):
    cohort_path=a.root/'cohort.json'
    all_rows=json.loads(cohort_path.read_text())['rows']
    rows=[r for r in all_rows if r['split']==a.phase]
    dev_docs={r['document_id'] for r in all_rows if r['split']=='development'}
    hold_docs={r['document_id'] for r in all_rows if r['split']=='confirmation'}
    if dev_docs & hold_docs:
        raise ValueError('Document leakage')
    if len({normalized(r['text']) for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate proposition text')
    spec=json.loads(a.spec.read_text())
    refs=json.loads(a.reference.read_text())['cases']
    for r in rows:
        ref=refs[r['id']]
        if ref['primary'] not in ref['acceptable_primary']:
            raise ValueError('Preferred reference must be acceptable')
        labels=ref['acceptable_primary']+ref['allowed_secondary']+ref.get('essential_secondary',[])
        if not set(labels).issubset(set(ROLES)|{'OTHER','UNCERTAIN'}):
            raise ValueError('Unknown reference role')
    if a.phase=='confirmation':
        if not a.selection or json.loads(a.selection.read_text())['spec_sha256']!=identity(a.spec)['sha256']:
            raise ValueError('Confirmation requires frozen prompt selection')
    jobs=[make_job(r,spec) for r in rows]
    models=ROOT.parent/'models'; runtime=ROOT.parent/'tools/kev-runtime'
    files={'cohort':cohort_path,'prompt':a.spec,'reference':a.reference,'runner':Path(__file__),
           'runtime':ROOT/'medrag/evidence_graph_runtime.py','kev_model':runtime/'kev/model.py',
           'adapter':models/'kev-9b/adapter_model.safetensors','head':models/'kev-9b/head.pt',
           'adapter_config':models/'kev-9b/adapter_config.json','base_config':models/'Qwen3.5-9B-Base/config.json'}
    for i,f in enumerate(sorted((models/'Qwen3.5-9B-Base').glob('*.safetensors'))):files[f'base_weights_{i}']=f
    contract={'version':'primary_secondary_roles_v1','prompt_version':spec['version'],'phase':a.phase,
              'change':spec['change'],'inputs':{k:identity(v) for k,v in files.items()},
              'rows':[r['id'] for r in rows], 'generation':'none; argmax decision probabilities',
              'gpu':1,'dtype':'bfloat16','batch_rows':a.batch_rows,'seed':'20261025-main-role',
              'scope':'official train corpus; proposition only; no QA options/answers/source-context',
              'max_development_versions':4,'selected_prompt':identity(a.selection) if a.selection else None,
              'hypothesis':'Direct dominant-function decision protects core roles despite imperfect auxiliary labels.',
              'criterion':'Qualitative: core preserved on most new cases, no recurring severe wrong-function family or source collapse; auxiliary errors do not alone veto main-label use. Quantitative agreement is supporting evidence, not human gold.',
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft')}}
    if a.preflight:
        sys.path.insert(0,str(runtime))
        from kev.api import SystemOneRequest, to_record
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(models/'kev-9b',local_files_only=True)
        for row in rows:
            for job in (make_job(row,spec),make_job(row,spec,'DEFINE')):
                rec,_=to_record(SystemOneRequest.model_validate(job['body']))
                if len(tok.encode(json.dumps(rec)))>7000:raise ValueError('Prompt budget exceeded')
        print(json.dumps({'preflight':'passed','n':len(rows),'gpu_loaded':False}));return
    out=a.root/spec['version']/a.phase; out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        h=ensure_contract(out/'manifest.json',contract); cache=DecisionCache(out/'cache',h)
        atomic_jsonl(out/'cohort.jsonl',rows)
        p=Progress(('contract', 'load KEV', 'primary decisions', 'secondary decisions', 'save review'))
        start=time.monotonic(); judge=None
        def emit(event,**data):
            if event=='stage':p.stage(**data)
            elif event=='update':p.update(data['amount'])
            else:p.complete(data['message'])
        def score(js,stage):
            nonlocal judge
            if any(cache.get(j) is None for j in js):
                if judge is None:
                    raise RuntimeError('Model was not loaded for missing cache')
                judge.score(js,stage,emit)
            else:
                total=sum(len(j['body']['questions']) for j in js)
                p.stage(stage,total,initial=total);p.complete('Compatible decisions cached')
        try:
            p.weights=[.2,5.,None,None,.2]
            p.log(f'[plan] {spec["version"]}/{a.phase}: {len(rows)} propositions -> primary -> conditional secondary; GPU1 only, no QA')
            p.log('[ETA] Prior six-role KEV92: 15-25s including load; changed two-pass prompts calibrate separately; qualitative review excluded')
            p.stage(0,1);p.update();p.complete('Frozen cohort/prompt/reference and model identities')
            p.stage(1,1)
            cached_primary=all(cache.get(j) is not None for j in jobs)
            sjobs=[make_job(r,spec,cache.get(j)['answers']['primary']['choice']) for r,j in zip(rows,jobs)] if cached_primary else []
            missing=not cached_primary or any(cache.get(j) is None for j in sjobs)
            if missing:
                if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Require CUDA_VISIBLE_DEVICES=1')
                from transformers.utils import logging as log
                log.set_verbosity_error();log.disable_progress_bar()
                judge=LocalDecisions(runtime,models/'kev-9b',models/'Qwen3.5-9B-Base',cache,a.batch_rows)
            p.update();p.complete('KEV loaded' if judge else 'Complete cached run; no GPU loaded')
            score(jobs,2)
            sjobs=[make_job(r,spec,cache.get(j)['answers']['primary']['choice']) for r,j in zip(rows,jobs)]
            score(sjobs,3)
            p.stage(4,1)
            preds=[{**r,'primary':cache.get(j)['answers']['primary'],'secondary':cache.get(s)['answers']}
                   for r,j,s in zip(rows,jobs,sjobs,strict=True)]
            report=evaluate(preds,refs)
            report.update(runtime_seconds=time.monotonic()-start,batch_parity=judge.parity if judge else None,
                          batch_fallback=judge.batch_disabled if judge else None,OOM_splits=judge.oom_splits if judge else None)
            atomic_jsonl(out/'primary_requests.jsonl',jobs);atomic_jsonl(out/'secondary_requests.jsonl',sjobs)
            atomic_jsonl(out/'predictions.jsonl',preds);atomic_json(out/'summary.json',report)
            atomic_jsonl(out/'review_items.jsonl',[{**r,'reference':refs[r['id']]} for r in preds])
            p.update();p.complete(f'complete={len(preds)}/{len(rows)} missing=0; {out}')
        except BaseException as error:
            done=sum(cache.get(j) is not None for j in jobs)
            p.log(f'[FAILED] stage={p.index+1}/5 primary={done}/{len(jobs)} secondary may be partial; durable={out}; same compatible command resumes; {error}')
            raise
        finally:p.close()
    print(json.dumps({'output':str(out),'n':len(rows),'primary_acceptable':report['primary_acceptable_n']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--phase',choices=('development','confirmation'),default='development')
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--batch-rows',type=int,default=48)
    parser.add_argument('--preflight',action='store_true')
    a=parser.parse_args()
    if a.prepare:prepare(a.root);return
    if not a.spec or not a.reference:parser.error('Require --spec and --reference')
    if not 6<=a.batch_rows<=96:parser.error('batch rows must be 6..96')
    run(a)


if __name__=='__main__':main()

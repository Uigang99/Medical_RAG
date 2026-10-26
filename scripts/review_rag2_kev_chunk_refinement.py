#!/usr/bin/env python3
"""Paired prompt comparison and version-blinded positive-edge audit sampling."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medrag.kev_chunk_relations import TASKS
from medrag.training.graph_input_data import atomic_json,atomic_jsonl,digest,sha_file


def read_rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def reference_metrics(refs,pred,relation):
    c=Counter();confusion=Counter()
    for r in refs:
        for task in TASKS:
            if TASKS[task][1]!=relation:continue
            truth=r['labels'][task];label=pred[r['pair_id']]['answers'][task]['choice']
            c['reviewed']+=1
            if truth=='UNCERTAIN':c['reference_uncertain']+=1;continue
            c['scored']+=1;c['reference_yes']+=truth=='YES';c['tp']+=truth==label=='YES'
            c['fp']+=truth=='NO' and label=='YES';c['fn']+=truth=='YES' and label!='YES'
            c['correct']+=truth==label;c['always_no_correct']+=truth=='NO'
            c['predicted_no']+=label=='NO';c['correct_no']+=label==truth=='NO'
            c['predicted_uncertain']+=label=='UNCERTAIN'
            confusion[truth+'->'+label]+=1
    def div(a,b):return a/b if b else None
    p=div(c['tp'],c['tp']+c['fp']);r=div(c['tp'],c['reference_yes'])
    return {'counts':dict(c),'confusion':dict(confusion),'precision':p,'recall':r,
            'f1':2*p*r/(p+r) if p is not None and r is not None and p+r else 0,
            'accuracy':div(c['correct'],c['scored']),'always_no_accuracy':div(c['always_no_correct'],c['scored']),
            'accuracy_delta':div(c['correct']-c['always_no_correct'],c['scored'])}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('sample','report'),required=True)
    ap.add_argument('--root',type=Path,default=ROOT/'results/rag2_kev_chunk_refinement_train12_v2')
    a=ap.parse_args();root=a.root
    c=json.loads((root/'cohort.json').read_text());pairs={p['pair_id']:p for p in c['pairs']}
    refs=read_rows(root/'blind_references.jsonl')
    frozen=json.loads((root/'frozen_reference_manifest.json').read_text())['contract']
    if sha_file(root/'blind_references.jsonl')!=frozen['reference_identity']['sha256']:raise ValueError('References changed')
    qsplit={q['case_id']:q['audit_split'] for q in c['questions']}
    pred={}
    for version in ('v1','v2'):
        records=read_rows(root/version/'predictions.jsonl')
        pred[version]={p['pair_id']:p for p in records}
        if len(records)!=len(pairs) or set(pred[version])!=set(pairs):raise ValueError('Incomplete or duplicate predictions')
    relations=list(dict.fromkeys(v[1] for v in TASKS.values()))
    report={'reference_type':'Frozen assistant provisional labels; not independent expert gold',
        'questions':12,'pairs':336,'blind_pairs':len(refs),'QA_evaluated':False,'scope':'Same official Train chunks and KEV; only prompt changed',
        'reference_comparison':{},'distributions':{},'paired_changes':{}}
    for split in ('all','development','confirmation'):
        subset=[r for r in refs if split=='all' or qsplit[r['case_id'].split('_')[0]]==split]
        report['reference_comparison'][split]={rel:{v:reference_metrics(subset,pred[v],rel) for v in pred} for rel in relations}
    changed=[];positives=defaultdict(dict)
    for pid,pair in pairs.items():
        for task in TASKS:
            rel=TASKS[task][1];audit_id=pair['case_id']+':'+task
            for v in pred:
                label=pred[v][pid]['answers'][task]['choice']
                report['distributions'].setdefault(v,{}).setdefault(rel,Counter())[label]+=1
                if label=='YES':positives[rel].setdefault(audit_id,{'audit_id':audit_id,'pair_id':pid,'case_id':pair['case_id'],'task':task,'versions':[]})['versions'].append(v)
            x,y=(pred[v][pid]['answers'][task]['choice'] for v in ('v1','v2'))
            report['paired_changes'].setdefault(rel,Counter())[x+'->'+y]+=1
            if x!=y:changed.append({'audit_id':audit_id,'pair_id':pid,'v1':x,'v2':y,'relation':rel})
    # Shared deterministic ordering encourages overlap without conditioning on correctness.
    # Twenty predicted positives per version/relation; preserve both halves when available.
    chosen={}
    for rel,items in positives.items():
        for v in pred:
            candidates=sorted((x for x in items.values() if v in x['versions']),key=lambda x:digest(['audit_v2',x['audit_id']]))
            selected=[]
            for split in ('development','confirmation'):
                selected.extend([x for x in candidates if qsplit[x['case_id'].split('_')[0]]==split][:10])
            keys={x['audit_id'] for x in selected}
            selected.extend([x for x in candidates if x['audit_id'] not in keys][:max(0,20-len(selected))])
            for x in selected:chosen.setdefault(x['audit_id'],dict(x,relation=rel,sampled_for=[]))['sampled_for'].append(v)
    sample=sorted(chosen.values(),key=lambda x:x['audit_id'])
    for x in sample:
        pair=pairs[x['pair_id']];x.update(A=pair['A'],B=pair['B'])
    fullpath=root/'positive_audit_sampling.jsonl'
    if fullpath.exists() and read_rows(fullpath)!=sample:raise ValueError('Audit sample changed')
    atomic_jsonl(fullpath,sample)
    # Review without model probabilities, version identity, or which version selected YES.
    atomic_jsonl(root/'positive_audit_blind_inputs.jsonl',[
        {k:v for k,v in x.items() if k not in ('versions','sampled_for')} for x in sample])
    atomic_jsonl(root/'changed_decisions.jsonl',changed)
    report['audit_union_size']=len(sample)
    report['audit_sample_counts']={rel:{v:sum(x['relation']==rel and v in x['sampled_for'] for x in sample) for v in pred} for rel in relations}
    if a.stage=='report':
        reviews=read_rows(root/'positive_audit_reviews.jsonl');byid={x['audit_id']:x for x in reviews}
        if len(byid)!=len(reviews) or set(byid)!={x['audit_id'] for x in sample}:raise ValueError('Incomplete/duplicate audit')
        if any(x['verdict'] not in ('CLEAR','BORDERLINE','UNSUPPORTED') for x in reviews):raise ValueError('Bad audit label')
        report['positive_audit']={}
        for rel in relations:
            report['positive_audit'][rel]={}
            for v in pred:
                selected=[x for x in sample if x['relation']==rel and v in x['sampled_for']]
                counts=Counter(byid[x['audit_id']]['verdict'] for x in selected);n=len(selected)
                precision=counts['CLEAR']/n if n else None
                recall=report['reference_comparison']['all'][rel][v]['recall']
                outcome=('insufficient_audit' if n<20 else 'fails_clear_precision' if precision<.85 else
                         'fails_reference_recall' if recall is None or recall<.70 else 'pilot_candidate_needs_larger_confirmation')
                report['positive_audit'][rel][v]={'n':n,'counts':dict(counts),'clear_fraction':precision,
                    'optimistic_fraction':(counts['CLEAR']+counts['BORDERLINE'])/n if n else None,'outcome':outcome}
        report['audit_limitations']='Same assistant; version-blinded but positives-only review; clustered small sample; no independent medical gold; not QA benefit.'
    atomic_json(root/'quality_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

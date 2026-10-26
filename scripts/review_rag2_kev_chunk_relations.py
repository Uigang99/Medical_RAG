#!/usr/bin/env python3
"""Sample positive edges and summarize explicitly provisional source-text reviews."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medrag.kev_chunk_relations import TASKS
from medrag.training.graph_input_data import atomic_json,atomic_jsonl,digest,sha_file


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def wilson(k,n):
    if not n: return None
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'results/rag2_kev_chunk_relations_test24_v1')
    parser.add_argument('--stage',choices=('sample','report'),required=True)
    a=parser.parse_args(); root=a.root
    cohort=json.loads((root/'cohort.json').read_text()); predictions=rows(root/'predictions.jsonl')
    refs=rows(root/'blind_references.jsonl')
    frozen=json.loads((root/'frozen_reference_manifest.json').read_text())['contract']
    if sha_file(root/'blind_references.jsonl')!=frozen['reference_identity']['sha256']:
        raise ValueError('Frozen references changed after KEV run')
    pairs={p['pair_id']:p for p in cohort['pairs']}; preds={p['pair_id']:p for p in predictions}
    if set(pairs)!=set(preds) or len(preds)!=len(predictions):
        raise ValueError('Incomplete/duplicate predictions')
    positives=defaultdict(list)
    for p in predictions:
        for task, answer in p['answers'].items():
            if answer['choice']=='YES':
                relation=TASKS[task][1]
                positives[relation].append({'audit_id':p['case_id']+':'+task,
                    'pair_id':p['pair_id'],'case_id':p['case_id'],'task':task,'relation':relation,
                    'probabilities':answer['probabilities'],'A':pairs[p['pair_id']]['A'],
                    'B':pairs[p['pair_id']]['B']})
    sample=[]
    for relation, items in sorted(positives.items()):
        sample.extend(sorted(items,key=lambda x:digest([2026102607,'positive_audit',x['audit_id']]))[:30])
    sample_path=root/'positive_audit_inputs.jsonl'
    if sample_path.exists() and rows(sample_path)!=sample:
        raise ValueError('Positive audit sample changed')
    atomic_jsonl(sample_path,sample)
    stats={}
    discrepancies=[]
    for relation in dict.fromkeys(v[1] for v in TASKS.values()):
        counts=Counter(); confusion=Counter()
        for ref in refs:
            for task in TASKS:
                if TASKS[task][1]!=relation: continue
                truth=ref['labels'][task]; pred=preds[ref['pair_id']]['answers'][task]['choice']
                counts['all_reviewed']+=1
                if truth=='UNCERTAIN':
                    counts['ambiguous_reference']+=1
                    continue
                counts['scored']+=1; counts['reference_yes']+=truth=='YES'
                counts['tp']+=truth=='YES' and pred=='YES'
                counts['fp']+=truth=='NO' and pred=='YES'
                counts['fn']+=truth=='YES' and pred!='YES'
                counts['exact']+=truth==pred; counts['always_no_exact']+=truth=='NO'
                confusion[truth+'->'+pred]+=1
                if truth!=pred:
                    discrepancies.append({'case_id':ref['case_id'],'task':task,'reference':truth,
                                          'prediction':pred,'note':ref['note']})
        n=counts['scored']; positive_n=counts['tp']+counts['fp']
        precision=counts['tp']/positive_n if positive_n else None
        recall=counts['tp']/counts['reference_yes'] if counts['reference_yes'] else None
        stats[relation]={'counts':dict(counts),'confusion':dict(confusion),'precision':precision,
                        'recall':recall,'exact_accuracy':counts['exact']/n if n else None,
                        'always_no_accuracy':counts['always_no_exact']/n if n else None,
                        'accuracy_delta':(counts['exact']-counts['always_no_exact'])/n if n else None,
                        'full_predicted_yes':len(positives[relation])}
    report={'reference_type':'assistant provisional, frozen before predictions; not independent clinical gold',
            'questions':len(cohort['questions']),'pairs':len(predictions),'blind_pairs':len(refs),
            'reference_agreement':stats,'positive_sample_counts':dict(Counter(x['relation'] for x in sample)),
            'QA_accuracy_measured':False,'thresholds':{'positive_precision':.85,'reference_recall':.70,'minimum_audit':20}}
    atomic_jsonl(root/'reference_disagreements.jsonl',discrepancies)
    if a.stage=='report':
        audits=rows(root/'positive_audit_reviews.jsonl')
        if len(audits)!=len(sample) or {x['audit_id'] for x in audits}!={x['audit_id'] for x in sample}:
            raise ValueError('Incomplete or duplicate qualitative audits')
        records={x['audit_id']:x for x in sample}; audited=defaultdict(Counter)
        for audit in audits:
            if audit['verdict'] not in ('CLEAR','BORDERLINE','UNSUPPORTED'):
                raise ValueError('Invalid qualitative audit verdict')
            audited[records[audit['audit_id']]['relation']][audit['verdict']]+=1
        report['positive_audit']={}
        for relation,counts in audited.items():
            n=sum(counts.values()); clear=counts['CLEAR']; upper=clear+counts['BORDERLINE']
            recall=stats[relation]['recall']
            outcome=('insufficient_positive_audit' if n<20 else 'fails_precision' if clear/n<.85
                     else 'insufficient_reference_positives' if stats[relation]['counts'].get('reference_yes',0)<10
                     else 'fails_recall' if recall is None or recall<.70 else 'candidate_only_not_QA_validated')
            report['positive_audit'][relation]={'n':n,'counts':dict(counts),'clear_fraction':clear/n,
                'optimistic_fraction':upper/n,'descriptive_wilson95':wilson(clear,n),'outcome':outcome,
                'uncertainty_note':'Edges share questions/documents; this item-level interval ignores clustering and reviewer error.'}
    atomic_json(root/'quality_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

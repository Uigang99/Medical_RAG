#!/usr/bin/env python3
"""Report the frozen, paired KEV27 chunk-relation confirmation without tuning."""
from collections import Counter, defaultdict
import json
from math import comb
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medrag.kev_chunk_relations import TASKS
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest
from review_rag2_kev_chunk_refinement import reference_metrics

BASE=ROOT/'results/rag2_kev27_prompt_quality_round2_v1'
RELATIONS=list(dict.fromkeys(v[1] for v in TASKS.values()))


def rows(path):
    with path.open() as f:return [json.loads(line) for line in f if line.strip()]


def macro(refs,pred):
    return sum(reference_metrics(refs,pred,rel)['f1'] for rel in RELATIONS)/len(RELATIONS)


def main(root=BASE):
    phase=root/'confirmation';cohort=json.loads((phase/'cohort.json').read_text())
    refs=rows(phase/'references.jsonl');pairs={p['pair_id']:p for p in cohort['pairs']}
    assert len(cohort['questions'])==12 and len(pairs)==336 and len(refs)==48
    pred={}
    for version in ('v2','v7'):
        rec=rows(phase/'v2_v7'/version/'predictions.jsonl')
        pred[version]={r['pair_id']:r for r in rec}
        assert len(rec)==len(pairs) and set(pred[version])==set(pairs)
    evaluated=[];transitions=Counter();byrel={}
    for rel in RELATIONS:
        metrics={v:reference_metrics(refs,pred[v],rel) for v in pred}
        reference_positive_cases={r['case_id'].split('_')[0] for r in refs
                                  for task in TASKS if TASKS[task][1]==rel and r['labels'][task]=='YES'}
        predicted_positive_cases={pairs[pid]['case_id'].split('_')[0] for pid,record in pred['v7'].items()
                                  for task in TASKS if TASKS[task][1]==rel
                                  and record['answers'][task]['choice']=='YES'}
        m=metrics['v7'];p=m['precision'];r=m['recall']
        count=m['counts'];enough=(count['reference_yes']>=10 and count['tp']+count['fp']>=10
                and len(reference_positive_cases)>=5 and len(predicted_positive_cases)>=5)
        verdict='insufficient_diversity' if not enough else 'pass' if p>=.80 and r>=.65 else 'fail'
        byrel[rel]={'metrics':metrics,'absolute_delta':{
            key:m[key]-metrics['v2'][key] if m[key] is not None and metrics['v2'][key] is not None else None
            for key in ('precision','recall','f1','accuracy')},
            'reference_positive_cases':len(reference_positive_cases),
            'predicted_positive_cases':len(predicted_positive_cases),
            'expansion_gate':verdict}
    for ref in refs:
        pair=pairs[ref['pair_id']]
        for task in TASKS:
            truth=ref['labels'][task]
            if truth=='UNCERTAIN':continue
            old=pred['v2'][ref['pair_id']]['answers'][task]['choice']
            new=pred['v7'][ref['pair_id']]['answers'][task]['choice']
            oldok=old==truth;newok=new==truth
            transitions[('C' if oldok else 'W')+'->'+('C' if newok else 'W')]+=1
            if old!=new or not newok:
                evaluated.append({'case_id':ref['case_id'],'task':task,'relation':TASKS[task][1],
                    'reference':truth,'v2':old,'v7':new,'note':ref['note'],
                    'A':pair['A']['text'],'B':pair['B']['text']})
    rng=random.Random(2026102802);questions=sorted({r['case_id'].split('_')[0] for r in refs});byq=defaultdict(list)
    for ref in refs:byq[ref['case_id'].split('_')[0]].append(ref)
    deltas=[]
    for _ in range(2000):
        sample=[r for q in rng.choices(questions,k=len(questions)) for r in byq[q]]
        deltas.append(macro(sample,pred['v7'])-macro(sample,pred['v2']))
    deltas.sort();ci=[deltas[50],deltas[1949]]
    total=sum(byrel[x]['metrics']['v7']['counts']['scored'] for x in RELATIONS)
    improved=transitions['W->C'];harmed=transitions['C->W'];discordant=improved+harmed
    paired_p=min(1.0,2*sum(comb(discordant,k) for k in range(min(improved,harmed)+1))/2**discordant)
    report={'scope':'Official Train12 new chunks; fixed reference before model; no QA or TEST evaluation',
        'questions':12,'pairs':336,'reviewed_pairs':48,'task_judgments':288,
        'scored_judgments':total,'excluded_reference_uncertain':288-total,
        'baseline':'v2','selected':'v7','relations':byrel,
        'macro_f1':{v:macro(refs,pred[v]) for v in pred},
        'macro_f1_absolute_delta':macro(refs,pred['v7'])-macro(refs,pred['v2']),
        'question_cluster_bootstrap_95ci_absolute_delta':ci,
        'paired_correctness_transitions':dict(transitions),
        'judgment_accuracy':{'v2':(transitions['C->C']+harmed)/total,
                             'v7':(transitions['C->C']+improved)/total},
        'paired_exact_mcnemar_p':paired_p,
        'all_relations_pass':all(x['expansion_gate']=='pass' for x in byrel.values()),
        'caveats':['Assistant provisional references, not medical-expert consensus',
                   'Small number of process-positive reference cases',
                   'Exact chunks can recur within corpus although sampled cohort chunks are disjoint',
                   'No MCQ accuracy or downstream RAG effect measured']}
    atomic_json(phase/'quality_report.json',report)
    atomic_jsonl(phase/'qualitative_review_queue.jsonl',evaluated)
    print(json.dumps({'macro_f1':report['macro_f1'],'delta':report['macro_f1_absolute_delta'],
                      'paired':report['paired_correctness_transitions'],
                      'expansion_gate':{x:v['expansion_gate'] for x,v in byrel.items()},
                      'report':str(phase/'quality_report.json')},ensure_ascii=False))


if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else BASE)

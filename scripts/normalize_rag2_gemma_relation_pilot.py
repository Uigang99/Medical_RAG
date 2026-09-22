#!/usr/bin/env python3
"""Conservative offline UMLS candidates plus surface-preserving relation keys."""
from __future__ import annotations

import json
import argparse
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from build_rag2_medical_kg_mvp import load_pipeline
from medrag.kg.biomedical_mvp import category_from_tuis

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/rag2_gemma_relation_pilot192_v1'
MODELS=('Qwen3.5-9B','gemma-4-31B-it')
os.environ.setdefault('SCISPACY_CACHE',str(ROOT.parent/'models/RAG2-Medical-KG-MVP/scispacy_cache'))
AUTO_TYPES={'condition','drug_or_chemical','procedure_or_intervention','anatomy',
            'organism','gene_or_protein','diagnostic_test','finding_or_outcome'}
TYPE_CATEGORY={'condition':{'DiseaseOrDisorder'},'drug_or_chemical':{'DrugOrChemical'},
               'procedure_or_intervention':{'ProcedureOrTreatment'},'anatomy':{'Anatomy'},
               'organism':{'Organism'},'gene_or_protein':{'GeneOrProtein'},
               'diagnostic_test':{'DiagnosticTest'},'finding_or_outcome':{'FindingOrSymptom'}}


def rows(path):
    with path.open() as f:
        for line in f:yield json.loads(line)


def save_rows(path,items):
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('w') as out:
        for item in items:out.write(json.dumps(item,ensure_ascii=False)+'\n')
    temp.replace(path)


def save(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def lexical(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    return ' '.join(re.sub(r'[^\w]+',' ',text).split())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root',type=Path,default=RESULT)
    args=parser.parse_args()
    result=args.result_root.resolve()
    paired=list(rows(result/'paired_review.jsonl'))
    if len(paired)!=192:raise RuntimeError('Paired generation output incomplete')
    normalized_path=result/'normalized_review_typeaware_v2.jsonl'
    if normalized_path.exists():
        print('[cached] normalized review already exists');return
    print('[normalization] loading existing offline scispaCy/UMLS index',flush=True)
    nlp,linker=load_pipeline('en_core_sci_sm','umls',0.85,3)
    kinds={}
    for row in paired:
        for model in MODELS:
            for e in row['models'][model]['entities']:
                kinds.setdefault(e['span'],set()).add(e['type'])
    surfaces=sorted(kinds)
    raw=linker.candidate_generator(surfaces,k=3)
    if len(raw)!=len(surfaces):raise RuntimeError('UMLS output count mismatch')
    linked={}
    for surface,candidates in zip(surfaces,raw):
        options=[]
        for candidate in candidates:
            cui=candidate.concept_id
            concept=linker.kb.cui_to_entity.get(cui)
            if concept is None:continue
            score=max(candidate.similarities,default=0.)
            options.append({'cui':cui,'name':concept.canonical_name,'score':float(score),
                            'types':list(concept.types),'exact_alias':lexical(surface) in {lexical(a) for a in candidate.aliases}})
        options.sort(key=lambda x:-x['score'])
        linked[surface]=options
    output=[];stats=Counter()
    for row in paired:
        models={}
        for model in MODELS:
            extracted=row['models'][model];ents=[];rels=[]
            for e in extracted['entities']:
                candidates=linked[e['span']]
                compatible=[c for c in candidates if category_from_tuis(c['types']) in TYPE_CATEGORY.get(e['type'],set())]
                top=compatible[0] if compatible else None
                runner=compatible[1]['score'] if len(compatible)>1 else 0.
                qualified=bool(top and e['type'] in AUTO_TYPES and top['score']>=.90 and
                               top['score']-runner>=.05 and top['exact_alias'])
                concept_id='UMLS:'+top['cui'] if qualified else None
                ents.append({**e,'surface':e['span'],'lexical_key':lexical(e['span']),
                             'concept_id':concept_id,'normalization':'strict_umls' if qualified else 'surface_only',
                             'umls_candidates':candidates})
                stats[(model,'entities')]+=1;stats[(model,'strict_umls')]+=qualified
            for r in extracted['relations']:
                pred=r['predicate'];lemma=' '.join(t.lemma_.casefold() for t in nlp(pred) if not t.is_punct)
                rels.append({**r,'predicate_surface':pred,'predicate_lexical_key':lemma,
                             'semantic_relation_type':None})
                stats[(model,'relations')]+=1
            models[model]={'entities':ents,'relations':rels}
        output.append({k:v for k,v in row.items() if k!='models'}|{'models':models})
    save_rows(normalized_path,output)
    summary={'version':'typeaware_v2','rows':len(output),'unique_entity_surfaces':len(surfaces),
             'counts':{'/'.join(k):v for k,v in stats.items()},
             'policy':'UMLS identity only for compatible coarse semantic types with score>=0.90, compatible-candidate margin>=0.05, and exact normalized alias. Otherwise retain original surface. Predicate lemma is only a lexical index, not a semantic relation type.',
             'quality_limit':'UMLS similarity and exact alias are not independent concept-linking gold; paired semantic audit remains required.'}
    save(result/'normalization_summary_typeaware_v2.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

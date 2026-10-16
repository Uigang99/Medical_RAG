#!/usr/bin/env python3
"""Scope adapters for the frozen Test method. No prompt/parser replacements.

Expected counts are measured once from completed upstream artifacts, then the
original strict validators run against those counts. Semantic methods remain
the imported implementations; neither Test files nor their outputs are changed.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medrag.train_kg_data import load, rows, write_json, write_rows, sha_file, VERSION


def event(**value):
    print('@@TRAIN_KG '+json.dumps(value),flush=True)


class EventProgress:
    """The parent owns the footer. Children send structured progress only."""
    def __init__(self,names):
        self.names=names; self.estimates=[None]*len(names); self.actual=[None]*len(names)
        self.index=1; self.total=1; self.done=0; self.initial=0; self.unit='item'
        self.started=self.stage_started=time.time(); self.last=0
    def plan(self,details): pass  # parent supplies the true Train scope
    def start(self,index,total,*,initial=0,unit='item',estimate=None):
        self.index=index; self.total=max(1,total); self.done=self.initial=initial; self.unit=unit
        self.stage_started=time.time(); self.render(force=True)
    def render(self,force=False):
        if force or time.time()-self.last>=2:
            event(phase=self.index,name=self.names[self.index-1],done=self.done,total=self.total,
                  initial=self.initial,unit=self.unit)
            self.last=time.time()
    def update(self,amount=1): self.done+=amount; self.render()
    def complete(self,detail):
        self.done=self.total; self.render(force=True)
        event(phase=self.index,complete=True,detail=detail,elapsed=time.time()-self.stage_started)
    def log(self,message): print(message,flush=True)
    def close(self,force_complete=False): pass


def cli(*args):
    sys.argv=[sys.argv[0],*map(str,args)]


def throughput_profile(module,stage):
    """Execution-only overrides; output limits, prompts and parsers unchanged."""
    if stage in ('step2','step3','step4'):
        module.BATCH_CHOICES=(8,16,32,64,128)
        if hasattr(module,'common'):
            module.common.BATCH_CHOICES=(8,16,32,64,128)
    batch={'propositions':64,'repair':32,'step2':128,'step3':128,'step4':64,'step5':128}.get(stage)
    if batch is None: return
    if stage in ('step2','step3','step4','step5'):
        name='contract' if stage=='step2' else 'experiment_contract'
        original=getattr(module,name)
        def profiled_contract(*args,**kwargs):
            value=original(*args,**kwargs)
            value['execution_profile']='h200-throughput'
            value['generation']['oom_policy']=f'halve current batch from {batch} down to 8 on confirmed OOM; restore {batch} after recovery shard'
            return value
        setattr(module,name,profiled_contract)
    if stage=='step5':
        sys.argv+=['--requested-batch-size',str(batch),'--gpu-memory-utilization','0.90']
        module.is_oom=lambda e: any(t in (type(e).__name__+': '+str(e)).lower()
            for t in ('outofmemoryerror','cuda out of memory','cuda error: out of memory','cudamalloc'))
    else:
        sys.argv+=['--batch-size',str(batch),'--gpu-memory-utilization','0.90']
        if stage in ('propositions','repair','step2','step3'):
            sys.argv+=['--max-num-batched-tokens',str(32768 if stage=='step3' else 16384)]


def contract_scope(module, function, scope):
    original=getattr(module,function)
    def wrapped(*args,**kwargs):
        result=original(*args,**kwargs)
        result['version']=VERSION+'_'+module.__name__
        result['purpose']='Apply the frozen Test method to independent official Train Top8 sources.'
        result['train_scope']=scope
        result['adapter']={'path':__file__,'sha256':sha_file(__file__)}
        if 'stage_contract' in result:
            result['stage_contract']['input']='Completed upstream official Train artifact; counts in scope'
        return result
    setattr(module,function,wrapped)


def completed(root):
    manifest=load(root/'manifest.json'); marker=load(root/'COMPLETE.json')
    if manifest['contract_sha256']!=marker['contract_sha256']:
        raise RuntimeError(f'Upstream completion contract differs: {root}')
    return load(root/'summary.json')


def step2(root):
    import extract_rag2_stage2_relation_discovery_full as m
    import extract_rag2_stage2_relation_discovery_union_full as u
    s=completed(root/'usable')
    m.EXPECTED_TOP8_PROPOSITIONS=u.EXPECTED_PROPOSITIONS=s['propositions']
    m.EXPECTED_TOP8_BY_SOURCE=u.EXPECTED_BY_SOURCE=s['by_source']
    m.MAX_MODEL_LEN=5120; m.MAX_OUTPUT_TOKENS=2048; m.TwoLineProgress=EventProgress
    def contract(*args):
        c=u.union_contract(*args)
        c['version']=VERSION+'_step2'; c['scope'].update(
            retrieval='official Train balanced4*8 rerank8',documents_audited=s['documents_audited'],
            documents_with_usable_propositions=s['documents_with_usable_propositions'])
        c['evaluation_limit']='Frozen pilot estimates only; no full Train semantic audit'
        c['generation'].pop('context_extension_reason',None)
        c['adapter_sha256']=sha_file(__file__)
        return c
    def aggregate(output,propositions,spans,ch,progress):
        # Reuse original validation/aggregation, but correct scope and empty Test segment.
        result=u.union_aggregate(output,propositions,spans,ch,progress)
        result['scope'].update(documents_audited=s['documents_audited'],
            documents_with_usable_propositions=s['documents_with_usable_propositions'])
        result['by_segment'].pop('union_minus_top8',None)
        write_json(output/'summary.json',result)
        return result
    m.contract=contract; m.aggregate=aggregate
    cli('--input',root/'usable/usable_propositions.jsonl','--input-manifest',root/'usable/manifest.json',
        '--output-dir',root/'step2','--batch-size','32','--max-num-batched-tokens','16384')
    return m


def step3(root):
    import extract_rag2_stage3_endpoint_binding_union_full as m
    s=completed(root/'step2'); d=s['structural_diagnostics']
    m.EXPECTED_PROPOSITIONS=d['propositions']; m.EXPECTED_ASSERTIONS=d['assertions']
    m.EXPECTED_DOCUMENTS=s['scope']['documents_with_usable_propositions']
    m.EXPECTED_BY_SOURCE={k:v.get('assertions',0) for k,v in s['by_source'].items()}
    m.EXPECTED_BY_SEGMENT={k:v.get('assertions',0) for k,v in s['by_segment'].items()}
    m.TwoLineProgress=EventProgress
    contract_scope(m,'experiment_contract',s['scope'])
    cli('--stage2-root',root/'step2','--output-dir',root/'step3')
    return m


def step4(root):
    import filter_rag2_stage4_risk_routed_entailment_union_full as m
    s=completed(root/'step3')
    m.EXPECTED_FRAMES=s['structural_diagnostics']['assertions']
    m.EXPECTED_DOCUMENTS=s['scope']['documents']; m.EXPECTED_PROPOSITIONS=s['scope']['propositions']
    counts=Counter(); hard=Counter(); risk=Counter(); segment=defaultdict(Counter)
    for r in rows(root/'step3/binding_outputs.jsonl'):
        route,reasons=m.route_record(r)
        counts[route]+=1; segment[r['segment']][route]+=1
        if route=='hard_defer': hard.update(reasons)
        elif route=='risk': risk.update(reasons)
    if sum(counts.values())!=m.EXPECTED_FRAMES: raise RuntimeError('Incomplete Stage3 binding file')
    m.EXPECTED_ROUTES={k:counts[k] for k in ('ordinary','hard_defer','risk')}
    m.EXPECTED_HARD_REASONS=dict(sorted(hard.items())); m.EXPECTED_RISK_REASONS=dict(sorted(risk.items()))
    m.EXPECTED_BY_SEGMENT={k:{r:v[r] for r in m.EXPECTED_ROUTES} for k,v in segment.items()}
    m.TwoLineProgress=EventProgress
    contract_scope(m,'experiment_contract',s['scope'])
    original_prepare=m.prepare_routes
    def prepare_routes(*args,**kwargs):
        marker=original_prepare(*args,**kwargs)
        for route in ('ordinary','hard_defer','risk'):
            marker['routes'].setdefault(route,0)
        return marker
    m.prepare_routes=prepare_routes
    cli('--stage3-root',root/'step3','--output-dir',root/'step4')
    return m


def step5(root,mode):
    import normalize_rag2_stage5_contextual_union_full as m
    s=completed(root/'step4'); m.EXPECTED_FRAMES=s['result']['kept']; m.VERSION=VERSION+'_step5'
    contract_scope(m,'experiment_contract',s['scope'])
    initial=0
    if mode=='verify' and (root/'step5/prepare_complete.json').exists():
        total=load(root/'step5/prepare_complete.json')['tasks']
        from argparse import Namespace
        for start in range(0,total,m.VERDICT_SHARD):
            stop=min(total,start+m.VERDICT_SHARD)
            if not m.shard_path(Namespace(output_dir=root/'step5'),start,stop).exists(): break
            initial=stop
        event(phase=1,name='verify: cache/model loading',done=initial,total=max(1,total),initial=initial,unit='task')
    def progress(label,done,total,started):
        event(phase=1,name=label,done=done,total=total,initial=initial,unit='item')
    m.progress=progress
    cli('--stage4-root',root/'step4','--output-dir',root/'step5','--mode',mode)
    return m


def step67(root):
    import build_rag2_stage6_7_surface_kg_union_full as m
    s=completed(root/'step5'); docs=set(); props=set(); count=0
    for r in rows(root/'step5/normalized_frames.jsonl'):
        count+=1; docs.add(r['document_id']); props.add(r['proposition_id'])
    if count!=s['scope']['frames']: raise RuntimeError('Incomplete Stage5 frames')
    m.EXPECTED={'frames':count,'mentions':s['scope']['mentions'],'documents':len(docs),'propositions':len(props)}
    m.EXPECTED_STAGE5_CONTRACT=load(root/'step5/manifest.json')['contract_sha256']
    m.VERSION=VERSION+'_step67'; m.TwoLineProgress=EventProgress
    contract_scope(m,'experiment_contract',m.EXPECTED)
    cli('--stage5-root',root/'step5','--output-root',root/'kg')
    return m


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--stage',required=True,choices=('cohort','retrieval','restore','propositions','repair','usable','step2','step3','step4','step5','kg','export'))
    ap.add_argument('--mode',default='verify',choices=('inventory','candidates','prepare','verify','aggregate'))
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--reduce-batch-after-oom',action='store_true')
    ap.add_argument('--profile',choices=('reference','h200-throughput'),default='reference')
    a=ap.parse_args(); root=a.root
    def report(done,total,unit): event(phase=1,name=a.stage,done=done,total=total,unit=unit,initial=0)
    if a.stage in ('cohort','retrieval','restore','usable','export'):
        from medrag import train_kg_data as data
        fn={'cohort':data.select_cohort,'retrieval':data.prepare_documents,'restore':data.restore_documents,
            'usable':data.select_usable,'export':data.export_question_views}[a.stage]
        if a.preflight_only: raise ValueError('Use top-level --preflight-only for preparation stages')
        fn(root,report); return
    if a.stage=='propositions':
        m=importlib.import_module('extract_rag2_restored_gemma_propositions')
        m.TwoLineProgress=EventProgress
        cli('--input',root/'restored/restored_chunks.jsonl','--source-scope',root/'restored/source_scope.json',
            '--output-dir',root/'propositions','--batch-size','32','--max-model-len','16384')
    elif a.stage=='repair':
        m=importlib.import_module('retry_rag2_restored_top8_gemma_propositions'); m.TwoLineProgress=EventProgress
        s=completed(root/'propositions')
        candidates,_=m.load_candidates(root/'propositions/document_propositions.jsonl')
        cli('--source-dir',root/'propositions','--output-dir',root/'repair','--batch-size','32',
            '--expected-source-documents',s['counts']['documents'],'--expected-candidates',len(candidates),
            '--contract-version',VERSION+'_repair')
    else:
        m={'step2':step2,'step3':step3,'step4':step4,'step5':lambda r:step5(r,a.mode),'kg':step67}[a.stage](root)
    if a.profile=='h200-throughput': throughput_profile(m,a.stage)
    if a.preflight_only:
        sys.argv.append('--prepare-only' if a.stage=='step2' else '--preflight-only')
    if a.reduce_batch_after_oom:
        if a.stage not in ('propositions','repair','step2','step3','step4'): raise ValueError('Unsupported OOM recovery stage')
        sys.argv.append('--reduce-batch-after-oom')
    m.main()


if __name__=='__main__': main()

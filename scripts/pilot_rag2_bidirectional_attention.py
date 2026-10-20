#!/usr/bin/env python3
"""Frozen TRAIN128: causal/bi-KG masks x zero/oracle; both successes and harms."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from scripts import pilot_rag2_attention_oracle as legacy
from scripts.pilot_rag2_attention_oracle_range import replay_check
from medrag.generation.bidirectional_proposition_attention import BidirectionalPropositionModel
from medrag.training.attention_oracle import score, optimize_step, loss_and_logits, outcome
from medrag.training.attention_oracle_range import new_state, observe, milestone
from medrag.training.bidirectional_attention_diagnostic import summarize
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, sha_file, identity, jsonl
from medrag.training.graph_input_progress import Progress
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.bidirectional_attention_runtime import calibrate, step_with_retry

STAGES=['validate TRAIN128','load frozen Llama','mask / gradient safety',
        'causal and bi zero','causal oracle','bidirectional oracle','paired diagnosis']
CODE=legacy.CODE+['scripts/pilot_rag2_attention_oracle_range.py','medrag/training/attention_oracle_range.py',
    'medrag/generation/bidirectional_proposition_attention.py','medrag/training/bidirectional_attention_diagnostic.py',
    'medrag/training/bidirectional_attention_runtime.py',
    'scripts/pilot_rag2_bidirectional_attention.py','scripts/run_rag2_bidirectional_attention_pilot.sh']


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=ROOT/'results/rag2_attention_identifiability_v1')
    p.add_argument('--prepared-root',type=Path,default=ROOT/'results/rag2_official_train_proposition_attention_v1')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/rag2_bidirectional_attention_pilot_v2_fast')
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--query-chunk',type=int,default=128)
    p.add_argument('--memory-fraction',type=float,default=.88)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args(argv)
    if a.steps<2 or a.steps%2 or a.query_chunk<1:p.error('Positive even --steps >=2 and positive chunk required')
    if not .5<=a.memory_fraction<=.92:p.error('--memory-fraction must be in [0.5,0.92]')
    return a


def preflight(a):
    for root in (a.source_root,a.prepared_root):
        if a.output_root.resolve()==root.resolve() or root.resolve() in a.output_root.resolve().parents:
            raise RuntimeError('Output must be separate from immutable sources')
    src=json.loads((a.source_root/'manifest.json').read_text())
    donor=json.loads((a.prepared_root/'manifest.json').read_text())
    complete=json.loads((a.prepared_root/'prepared/COMPLETE.json').read_text())
    if digest(src['contract'])!=src['contract_sha256'] or digest(donor['contract'])!=donor['contract_sha256']:
        raise RuntimeError('Corrupt source manifest')
    if src['contract']['source_manifest']!=donor or complete['contract_sha256']!=donor['contract_sha256']:
        raise RuntimeError('Prepared input contract changed')
    rp=a.prepared_root/'prepared/rows.jsonl'; rh=sha_file(rp)
    if rh!=complete['rows_sha256'] or rh!=src['contract']['prepared_sha256']:
        raise RuntimeError('Prepared token cache changed')
    if donor['contract'].get('lora') is not None:raise RuntimeError('Require pristine Llama without LoRA')
    for meta in donor['contract']['inputs']['model_files'].values():
        actual=identity(meta['path'],full_hash='sha256' in meta)
        if any(actual.get(k)!=v for k,v in meta.items()):raise RuntimeError('Model/tokenizer changed: '+meta['path'])
    source_records=list(jsonl(a.source_root/'capability_interventions.jsonl'))
    prior={r['sample_id']:r for r in source_records}
    if len(source_records)!=128 or len(prior)!=128 or sum(r['base']['correct'] for r in source_records)!=96:
        raise RuntimeError('Require identical full capability128: original 96 correct and 32 wrong')
    if any(r['contract_sha256']!=src['contract_sha256'] for r in source_records):raise RuntimeError('Source record hash mismatch')
    counts=Counter(); rows=[]
    for row in jsonl(rp):
        counts[row['split']]+=1
        if row['sample_id'] not in prior:continue
        row=dict(row,diagnostic_split='capability')
        if row['split']!='train' or digest(row)!=prior[row['sample_id']]['row_sha256']:
            raise RuntimeError('Require exact original official-TRAIN question/tokens/layout')
        legacy.validate_layout(row)
        if len(set(row['choice_token_ids']))!=4 or not 0<=row['gold_index']<4:raise RuntimeError('Invalid option labels')
        rows.append(row)
    if counts!={'train':9000,'validation':1000} or len(rows)!=128:raise RuntimeError('Incomplete official data')
    rows.sort(key=lambda r:legacy.stable_key(314159,r['sample_id']))
    contract=dict(version='bidirectional_attention_pilot_v2_fast',source_hash=src['contract_sha256'],
        source_predictions_sha256=sha_file(a.source_root/'capability_interventions.jsonl'),prepared_sha256=rh,
        model=src['contract']['model'],model_files=donor['contract']['inputs']['model_files'],
        settings=dict(steps=a.steps,learning_rate=.1,bound=8.,query_chunk=a.query_chunk,seed=314159,batch=1,
            cache_frozen_prefix=True,cache_static_bidirectional_mask=True,memory_fraction=a.memory_fraction,
            checkpointing='Choose faster equivalent path on longest prompt separately per mask; OOM rolls back current step.',
            numerical_backend='Same MATH kernel and query chunk as v1; no fused-mask backend substitution'),
        cohort=[dict(sample_id=r['sample_id'],dataset=r['dataset'],row_sha256=digest(r),tokens=len(r['input_ids']),
            pairs=len(r['pairs']),source_correct=prior[r['sample_id']]['base']['correct']) for r in rows],
        code={p:sha_file(ROOT/p) for p in CODE},
        environment=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),
        scientific=dict(
            hypothesis='Opening KG-linked future evidence enables useful joint control unavailable under causal masks.',
            conditions=['causal zero','KG bidirectional zero','causal oracle','KG bidirectional oracle'],
            scope='All original capability TRAIN128, including original correct96 and wrong32. No Val1000/Test6545.',
            input='Exact original cached prompt/Top8 propositions, KG pairs, question/options and ordering. No retrieval or label in evidence.',
            mask='Original causal mask OR directed KG adjacency between proposition token blocks, ONLY at controlled layers. No other future keys.',
            grouping=[[i+1 for i in g] for g in legacy.GROUPS],
            parameters='Independent scalar per direction and group, shared across 4 layers and all heads. Llama frozen.',
            roles='Original S/S S/O O/S O/O retained; reverse roles are swapped in exported metadata. No relation-meaning classifier.',
            initialization='Both oracle arms start at zero with fresh Adam; no previously optimized table reused.',
            loss='One four-choice CE for the whole question, all available directions optimized jointly with gold.',
            selection='Best CE within each arm including own zero, plus raw endpoints at halfway/final. No causal fallback for the bidirectional arm.',
            primary='Bidirectional oracle minus causal oracle at equal updates; report C->W, W->C, net accuracy and CE.',
            secondary='Opening-only and within-mask optimization effects; original-correct/original-wrong/graphless and dataset breakdown.',
            screen='Exploratory: >=3 net extra correct /128 AND mean CE improvement >=.02 versus matched causal oracle. Not confirmation.',
            stop='Fixed cohort, one zero start, bound8, configured updates. All arms run; no adaptive budget expansion.',
            limitation='Bidirectional changes mask and doubles available directional parameters. No generalization/KG-specific causal claim.',
            numerical='BF16 frozen target, FP32 bias table/output head; controlled attention uses MATH with and without gradients.'))
    h=digest(contract);mp=a.output_root/'manifest.json'
    if mp.exists() and legacy.read_json(mp,h)['contract']!=contract:raise RuntimeError('Use a new output root for different contracts')
    parent=a.output_root
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<2*1024**3:raise RuntimeError('Require >=2GiB for durable outputs')
    return rows,prior,contract,h


def safety(w,row,pad):
    # Same Llama weights and original backend, independently built causal forward.
    ref=legacy.PropositionAttentionModel(w.llm,legacy.OracleBiasTable(w.llm.config.hidden_size,row['pairs']).to(w.llm.device),
        attention_layers=w.attention_layers,gradient_checkpointing=True,query_chunk=w.query_chunk).eval()
    from medrag.generation.bidirectional_proposition_attention import install
    install()
    checks={}
    for val in (0.,.25):
        w.configure(row,False)
        fixed=torch.full_like(w.controller.values,val)
        w.controller.assign(fixed);ref.controller.assign(fixed)
        expected=score(ref,row,pad);actual=score(w,row,pad)
        replay_check(actual,expected,'new versus legacy causal')
    for mode in (False,True):
        w.configure(row,mode)
        t=time.monotonic();first=score(w,row,pad); forward=time.monotonic()-t
        replay_check(score(w,row,pad),first,'repeat')
        # Check forward parity also with nonzero biases, not only at the origin.
        w.controller.assign(torch.full_like(w.controller.values,.25));expected=score(w,row,pad)
        w.train()
        with training_attention_backend():loss,logits=loss_and_logits(w,row,pad)
        replay_check(outcome(loss,logits,row['gold_index']),expected,'gradient/no-gradient')
        del loss,logits
        opt=torch.optim.Adam([w.controller.values],lr=.1)
        t=time.monotonic();_,_,diagnostic=optimize_step(w,row,pad,opt,8.);backward=time.monotonic()-t
        n=len(row['pairs']);g=w.controller.values.grad
        checks['bidirectional' if mode else 'causal']=dict(forward_seconds=forward,update_seconds=backward,
            repeat_and_nonzero_gradient_parity=True,forward_gradient_norm=float(g[:n].norm()),
            reverse_gradient_norm=float(g[n:].norm()) if mode else 0.,**diagnostic)
    checks.update(target_frozen=all(not p.requires_grad for p in w.llm.parameters()),causal_legacy_parity=True,
                  gpu=torch.cuda.get_device_name(0),peak_vram_bytes=torch.cuda.max_memory_allocated())
    return checks


def paths(a,row,mode):
    d=legacy.qdir(a,row)/mode
    return d/'last.pt',d/'result.json'


def measure_zero(a,w,rows,prior,pad,h,progress):
    cached=sum((legacy.qdir(a,r)/'zero.json').exists() for r in rows)
    progress.stage(3,len(rows)*2,initial=cached*2,unit='forward',kind='validation')
    result={}
    for row in rows:
        path=legacy.qdir(a,row)/'zero.json';i=row['sample_id']
        if path.exists():result[i]=legacy.read_json(path,h);continue
        values={}
        for mode in ('causal','bidirectional'):
            w.configure(row,mode=='bidirectional');values[mode]=score(w,row,pad)
        rec=dict(contract_sha256=h,sample_id=i,source_initially_correct=prior[i]['base']['correct'],**values)
        atomic_json(path,rec);result[i]=rec;progress.update(2)
    progress.complete(f'Both masks scored without gold-selected fallback; {a.output_root}/questions')
    return result


def optimize(a,w,row,zero,pad,h,mode,progress):
    ck,final=paths(a,row,mode)
    if final.exists():return legacy.read_json(final,h)
    w.configure(row,mode=='bidirectional');n=len(row['pairs'])
    seed=dict(values=w.controller.values.detach().cpu().tolist(),best=zero[mode])
    opt=torch.optim.Adam([w.controller.values],lr=.1)
    if ck.exists():
        state=legacy.read_tensor(ck,h)
        if state['sample_id']!=row['sample_id'] or state['mode']!=mode or state['ordered_pairs']!=w.ordered_pairs:
            raise RuntimeError('Checkpoint question/direction mismatch')
        w.controller.assign(state['values']);opt.load_state_dict(state['optimizer'])
    else:
        state=new_state(h,row,seed,8.)
        state.update(mode=mode,ordered_pairs=w.ordered_pairs,initial=zero[mode],updates=0,
            source_initially_correct=zero['source_initially_correct'])
        if n:milestone(state,zero[mode],w.controller.values.detach().cpu(),0)
    if not n:
        state.update(step=a.steps,complete=True)
        for step in (0,a.steps//2,a.steps):
            state['milestones'][str(step)]=dict(step=step,best=zero[mode],endpoint=zero[mode],
                best_values=[],best_step=0,available_correct=zero[mode]['correct'],
                endpoint_saturation_fraction=0.,best_saturation_fraction=0.)
        state['endpoint_values']=[]
    else:
        if not state['complete']:
            for step in range(state['step'],a.steps):
                result,before,diag=step_with_retry(w,row,pad,opt,8.,getattr(progress,'log',None))
                observe(state,result,before,step)
                g=w.controller.values.grad
                diag.update(forward_gradient_norm=float(g[:n].norm()),
                            reverse_gradient_norm=float(g[n:].norm()) if mode=='bidirectional' else 0.)
                state['history'].append(dict(step_before_update=step,score_before_update=result,**diag))
                state['step']=step+1;state['updates']=step+1
                if state['step'] in (a.steps//2,a.steps):
                    milestone(state,score(w,row,pad),w.controller.values.detach().cpu(),state['step'])
                if state['step']%5==0 or state['step'] in (a.steps//2,a.steps):
                    state.update(values=w.controller.values.detach().cpu().clone(),optimizer=opt.state_dict())
                    legacy.save_tensor(ck,state)
                progress.update()
            state.update(complete=True,values=w.controller.values.detach().cpu().clone(),optimizer=opt.state_dict())
            legacy.save_tensor(ck,state)
        state['endpoint_values']=state['values'].tolist()
        w.controller.assign(torch.tensor(state['best_values']))
        replay_check(score(w,row,pad),state['best'],'selected best table')
    record={k:v for k,v in state.items() if k not in ('values','optimizer')}
    record['ordered_pair_roles']=row['pair_roles']+([[v[0],v[2],v[1],v[3]] for v in row['pair_roles']] if mode=='bidirectional' else [])
    atomic_json(final,record)
    return record


def cached_updates(a,rows,mode,h):
    total=0
    for r in rows:
        if not r['pairs']:continue
        ck,final=paths(a,r,mode)
        if final.exists():
            rec=legacy.read_json(final,h)
            if not rec['complete'] or rec['step']!=a.steps:raise RuntimeError('Incomplete final record')
            total+=rec['updates']
        elif ck.exists():total+=legacy.read_tensor(ck,h)['updates']
    return total


def run(a,rows,prior,c,h,p):
    path=a.output_root/'manifest.json'
    if not path.exists():
        git={k:subprocess.check_output(['git',*v],cwd=ROOT,text=True).strip() for k,v in
             [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current'])]}
        atomic_json(path,dict(contract=c,contract_sha256=h,git=git))
    atomic_jsonl(a.output_root/'cohort.jsonl',c['cohort']);p.complete(f'Exact TRAIN128 contract; {path}')
    final=a.output_root/'SUMMARY.json'
    if final.exists():return legacy.read_json(final,h)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available():
        raise RuntimeError('GPU1 must be available on the server; no model allocated')
    p.stage(1,1)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error();logging.disable_progress_bar()
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(c['settings']['seed'])
    llm=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    w=BidirectionalPropositionModel(llm,rows[0],query_chunk=a.query_chunk,cache_prefix=True)
    p.complete('Frozen pristine Llama; original text; MATH on controlled layers')
    p.stage(2,1)
    check=safety(w,max((r for r in rows if r['pairs']),key=lambda r:len(r['input_ids'])),pad)
    check['contract_sha256']=h;atomic_json(a.output_root/'SAFETY.json',check)
    runtime=calibrate(w,max((r for r in rows if r['pairs']),key=lambda r:len(r['input_ids'])),
                      pad,a.memory_fraction,p.log)
    runtime['contract_sha256']=h
    # On resume retain earlier calibration records as well as the latest one.
    runtime_path=a.output_root/'RUNTIME.json'
    earlier=legacy.read_json(runtime_path,h)['sessions'] if runtime_path.exists() else []
    atomic_json(runtime_path,dict(contract_sha256=h,sessions=earlier+[runtime]))
    n=sum(bool(r['pairs']) for r in rows)
    p.weights[3]=len(rows)*sum(runtime['modes'][x]['selected']['forward_seconds'] for x in ('causal','bidirectional'))
    for j,mode in enumerate(('causal','bidirectional'),4):
        selected=runtime['modes'][mode]['selected']
        p.weights[j]=n*(a.steps*selected['update_seconds']+3*selected['forward_seconds'])
    p.weights[6]=2.
    p.complete('Legacy parity + cached/non-checkpointed loss/gradient/Adam parity; fastest memory-safe execution selected; RUNTIME.json')
    zero=measure_zero(a,w,rows,prior,pad,h,p);records={r['sample_id']:{} for r in rows}
    for j,mode in enumerate(('causal','bidirectional'),4):
        p.stage(j,n*a.steps,initial=cached_updates(a,rows,mode,h),unit='update',kind='train')
        for row in rows:
            p.names[j]=f'{mode} {row["dataset"]} oracle'
            records[row['sample_id']][mode]=optimize(a,w,row,zero[row['sample_id']],pad,h,mode,p)
        p.complete(f'{mode}: zero initialization, {a.steps} updates/question, graphless retained unchanged; {a.output_root}/questions')
    p.stage(6,1);report=summarize(rows,records,a.steps,h)
    report['execution_runtime']=runtime
    atomic_jsonl(a.output_root/'per_question.jsonl',report['per_question']);atomic_json(final,report)
    p.complete(f'Four conditions, paired gains/harms and raw endpoints; {final}')
    return report


def main(argv=None):
    a=arguments(argv);torch.set_num_threads(8);p=None
    try:
        if not a.preflight_only:p=Progress(list(STAGES));p.stage(0,1)
        rows,prior,c,h=preflight(a)
        counts=dict(questions=len(rows),datasets=dict(Counter(r['dataset'] for r in rows)),
            originally_correct=96,originally_wrong=32,graphless=sum(not r['pairs'] for r in rows),
            causal_pairs=sum(len(r['pairs']) for r in rows),zero_forwards=2*len(rows),
            optimizer_updates=sum(bool(r['pairs']) for r in rows)*2*a.steps,
            max_tokens=max(len(r['input_ids']) for r in rows),
            cached_updates={m:cached_updates(a,rows,m,h) for m in ('causal','bidirectional')})
        if a.preflight_only:
            print(json.dumps(dict(status='preflight_ok',contract_sha256=h,**counts,model_loaded=False,
                gpu_job_started=False,output=str(a.output_root)),indent=2));return
        p.log(f'[plan] frozen TRAIN128 -> load -> safety -> causal/bi zero -> causal oracle -> bi oracle -> gains/harms; {counts}')
        p.log('[scope] Gold-conditioned diagnosis, not learned routing or final TEST. Same zero initialization and steps, bound8, independent directional biases. No cross-mask no-op fallback.')
        p.log('[runtime] Cache unchanged layers 0..11 and static masks; use spare VRAM to avoid recomputation. Exact v1 query chunk/MATH/optimizer preserved; fastest parity-passing path selected per mask. OOM rollback is question-local.')
        p.log('[ETA] Old unoptimized estimate 3-5h; provisional 2-4h if retained activations fit H200. No fast H200 measurement yet; startup records matched old/new timings and VRAM in RUNTIME.json, then recalibrates ETA. Resume every 5 updates.')
        a.output_root.mkdir(parents=True,exist_ok=True)
        with (a.output_root/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            report=run(a,rows,prior,c,h,p)
        if p.index<6:
            for j in range(p.index+1,7):p.stage(j,1,initial=1);p.complete('cached complete diagnostic')
        p.log(f'[result] {report["status"]}; exploratory screen={report["screen_passed"]}; {a.output_root}/SUMMARY.json; NOT generalization accuracy')
    except Exception as e:
        if p:p.log(f'[FAILED] stage={p.index+1}/7 item={p.done}/{p.total} remaining={p.total-p.done}; {type(e).__name__}: {e}; durable={a.output_root}; identical command resumes completed records and optimizer checkpoints; source artifacts unchanged')
        raise
    finally:
        if p:p.close()


if __name__=='__main__':main()

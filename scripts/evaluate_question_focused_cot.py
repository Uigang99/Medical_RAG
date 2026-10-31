#!/usr/bin/env python3
"""One frozen short-prompt diagnostic comparison; no direct-answer rerun."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import evaluate_question_only_cot_baselines as matched
from medrag.generation.question_focused_cot import (
    NO_RAG_SYSTEM, RAG_SYSTEM, QuestionFocusedCoT, QuestionFocusedPCED, baseline_prompt, pced_prompt)
from medrag.generation.question_only_pced import kv_bytes_per_question
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, jsonl, sha_file
from medrag.training.graph_input_progress import Progress

VERSION = 'question_focused_cot_test6545_v1'
ARMS = ('no_rag_cot', 'base_rag_cot', 'pced_cot')


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pced-root',type=Path,default=ROOT/'results/question_only_pced_medcpt_two_modes_v1')
    p.add_argument('--previous-root',type=Path,default=ROOT/'results/question_only_cot_baselines_pced_matched_v1')
    p.add_argument('--output-root',type=Path,default=ROOT/'results'/VERSION)
    p.add_argument('--max-batch-questions',type=int,default=64)
    p.add_argument('--pced-batch-questions',type=int,default=32)
    p.add_argument('--memory-fraction',type=float,default=.88)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args(argv)
    if min(a.max_batch_questions,a.pced_batch_questions)<1 or a.limit<0 or not 0<a.memory_fraction<.98:
        p.error('Invalid runtime/cohort arguments')
    for key in ('pced_root','previous_root','output_root'):setattr(a,key,getattr(a,key).resolve())
    if a.limit and a.output_root==(ROOT/'results'/VERSION).resolve():
        a.output_root=a.output_root.with_name(VERSION+f'_smoke{a.limit}')
    for old in (a.pced_root,a.previous_root):
        if old==a.output_root or old in a.output_root.parents or a.output_root in old.parents:
            p.error('Output must be separate from historical results')
    return a


def load_previous(a,rows,pced):
    m=json.loads((a.previous_root/'manifest.json').read_text());c=m['contract']
    if digest(c)!=m['contract_sha256'] or c['version']!=matched.VERSION:
        raise RuntimeError('Invalid historical CoT manifest')
    for p,h in c['code_sha256'].items():
        if sha_file(ROOT/p)!=h:raise RuntimeError(f'Historical CoT source changed: {p}')
    expected=set(c['sample_ids']);selected={r['sample_id']:r for r in rows};results={'pced_cot':pced}
    for arm in matched.ARMS:
        seen=set();out={}
        for p in sorted((a.previous_root/arm/'shards').glob('*.jsonl')):
            for v in jsonl(p):
                sid=v['sample_id']
                if sid in seen or sid not in expected or v['contract_sha256']!=m['contract_sha256']:
                    raise RuntimeError(f'Historical shard mismatch: {p}/{sid}')
                seen.add(sid)
                if sid in selected:
                    matched.validate_record(selected[sid],v,arm,m['contract_sha256']);out[sid]=v
        if seen!=expected or set(out)!=set(selected):raise RuntimeError('Historical CoT is incomplete')
        results[arm]=out
    return m,results


def validate_record(row,v,arm,h):
    if arm!='pced_cot':return matched.validate_record(row,v,arm,h)
    if (v.get('sample_id')!=row['sample_id'] or v.get('dataset')!=row['dataset']
        or v.get('arm')!=arm or v.get('mode')!='rationale' or v.get('contract_sha256')!=h
        or v.get('document_ids')!=[d['corpus_id'] for d in row['documents']]
        or v.get('priors')!=row['priors'] or v.get('answer') not in ('A','B','C','D')
        or v.get('finish_reason') not in ('natural_marker','eos','token_budget')
        or len(v.get('expert_ranks',[]))!=len(v.get('rationale_token_ids',[]))
        or not math.isfinite(v.get('beta',float('nan')))):
        raise RuntimeError(f'Invalid PCED shard: {row["sample_id"]}')


def load_done(a,rows,arm,h):
    expected={r['sample_id']:r for r in rows};out={}
    for p in sorted((a.output_root/arm/'shards').glob('*.jsonl')):
        for v in jsonl(p):
            sid=v['sample_id']
            if sid not in expected or sid in out:raise RuntimeError(f'Duplicate/unexpected result: {sid}')
            validate_record(expected[sid],v,arm,h);out[sid]=v
    return out


def prompt_changes(rows,gold,old,new,resamples=5000):
    import numpy as np
    groups=[np.array([i for i,r in enumerate(rows) if matched.group(r['dataset'])==g]) for g in matched.GROUPS]
    before=np.array([[old[a][r['sample_id']]['answer']==gold[r['sample_id']] for a in ARMS] for r in rows])
    after=np.array([[new[a][r['sample_id']]['answer']==gold[r['sample_id']] for a in ARMS] for r in rows])
    change=after.astype(float)-before.astype(float);rng=np.random.default_rng(42)
    samples=np.array([np.mean([change[ix[rng.integers(len(ix),size=len(ix))]].mean(0) for ix in groups],axis=0) for _ in range(resamples)])
    out={}
    for j,arm in enumerate(ARMS):
        transitions={f'{a}->{b}':int(((before[:,j]==(a=='C'))&(after[:,j]==(b=='C'))).sum()) for a in ('C','W') for b in ('C','W')}
        out[arm]={'n':len(rows),'old_macro3':float(np.mean([before[ix,j].mean() for ix in groups])),
            'new_macro3':float(np.mean([after[ix,j].mean() for ix in groups])),
            'macro3_delta':float(np.mean([change[ix,j].mean() for ix in groups])),
            'macro3_delta_95_ci':np.quantile(samples[:,j],[.025,.975]).tolist(),
            'micro_delta':float(change[:,j].mean()),'transitions':transitions,
            'transition_rates_all_questions':{k:v/len(rows) for k,v in transitions.items()}}
    return out


def run(a):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error()
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=Progress(['CPU contracts/prompts','load frozen Llama','No-RAG CoT','Base-RAG CoT','PCED CoT','Macro-3 comparison'])
        p.weights=[180.,20.,436.3,1917.2,3697.9,5.]
        stage='CPU contracts/prompts';completed={}
        try:
            p.log('[plan] frozen Top-8 -> short prompts -> No-RAG/Base-RAG/PCED CoT -> old/new Macro-3; direct results untouched; GPU1')
            p.log('[scope] One fixed prompt candidate; TEST-informed diagnostic re-evaluation, not independent held-out validation. No retrieval/training or prompt search.')
            p.log('[ETA] Historical H200 generation: No-RAG 7.3m, Base-RAG 32.0m, PCED 61.6m for 6545; provisional total 1.5-2.5h. New prompt lengths/output rates recalibrate each arm.')
            p.stage(0,1)
            ref,cfg=matched.verify_reference(a)
            tok=AutoTokenizer.from_pretrained(cfg.model,local_files_only=True,padding_side='left')
            if tok.pad_token_id is None:tok.pad_token=tok.eos_token
            rows,gold,labels,counts,old_pced=matched.prepare(a,cfg,ref,tok)
            old_manifest,previous=load_previous(a,rows,old_pced)
            if old_manifest['contract']['reference_contract_sha256']!=ref['contract_sha256']:
                raise RuntimeError('Historical baselines and PCED refer to different experiments')
            reserve=cfg.max_rationale_tokens+len(tok.encode('\nFinal answer:',add_special_tokens=False))+1
            for r in rows:
                prompts={arm:[baseline_prompt(tok,r,arm)] for arm in matched.ARMS}
                prompts['pced_cot']=[pced_prompt(tok,r,d) for d in [None,*r['documents']]]
                r['short_lengths']={arm:max(len(tok.encode(s,add_special_tokens=False)) for s in texts) for arm,texts in prompts.items()}
                if max(r['short_lengths'].values())+reserve>cfg.max_model_len:
                    raise RuntimeError(f'Context overflow, no truncation allowed: {r["sample_id"]}')
            for arm,texts in prompts.items():
                # Same final chat boundary for every row; old label-token
                # stability was already checked for every row above.
                for prompt in texts:
                    prefix=tok.encode(prompt+'\nFinal answer:',add_special_tokens=False)
                    for label,tid in zip('ABCD',labels):
                        if tok.encode(prompt+'\nFinal answer: '+label,add_special_tokens=False)!=prefix+[tid]:raise RuntimeError('Label token mismatch')
            code=[Path(__file__),ROOT/'medrag/generation/question_focused_cot.py',Path(matched.__file__)]
            c={'version':VERSION,'reference_pced_sha256':ref['contract_sha256'],'reference_cot_sha256':old_manifest['contract_sha256'],
                'previous_predictions_sha256':digest(previous),'sample_ids':[r['sample_id'] for r in rows],
                'model':str(cfg.model),'seed':42,'split':'TEST-informed diagnostic; fixed candidate, not independent final validation',
                'systems':{'no_rag_and_amateur':NO_RAG_SYSTEM,'base_rag_and_experts':RAG_SYSTEM},
                'generation':old_manifest['contract']['generation'],'pced':{k:ref['contract'][k] for k in ['beta','gamma','prior','score']},
                'changed':'system prompts only; No-RAG adjustment also applies to amateur; beta recalculated by unchanged rule',
                'fixed':'same model/Top-8/text/order/greedy/token budget/EOS/label decoder; direct arms untouched',
                'hypothesis':'A short task/evidence role distinction reduces retrieval-induced CoT errors without changing decoding',
                'arms':list(ARMS),'primary':'Macro-3 (MedMCQA, MedQA, pooled medical MMLU); primary contrast new PCED minus new Base-RAG',
                'diagnostic_contrasts':'All paired new-vs-old prompt deltas; no prompt selection on this TEST run',
                'secondary':'C-to-W/W-to-C, micro, dataset accuracy, termination, tokens, beta and winner traces',
                'bootstrap':{'seed':42,'resamples':5000,'strata':list(matched.GROUPS)},
                'runtime':{'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'baseline_batch':a.max_batch_questions,'pced_batch':a.pced_batch_questions,
                    'memory_fraction':a.memory_fraction,'packages':ref['contract']['runtime']['packages']},
                'code_sha256':{str(path.relative_to(ROOT)):sha_file(path) for path in code},
                'git_at_creation':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
            manifest=a.output_root/'manifest.json'
            if manifest.exists():c['git_at_creation']=json.loads(manifest.read_text())['contract']['git_at_creation']
            h=ensure_contract(manifest,c);results={arm:load_done(a,rows,arm,h) for arm in ARMS}
            completed={arm:len(v) for arm,v in results.items()}
            atomic_json(a.output_root/'preflight.json',{'n':len(rows),'datasets':counts,'label_ids':labels,'max_prompt_tokens':{a:max(r['short_lengths'][a] for r in rows) for a in ARMS},'cached':completed,'previous_results_complete':True})
            p.complete(f'N={len(rows)}; cached={completed}; unchanged references; {a.output_root}')
            for i,arm in enumerate(ARMS,2):p.weights[i]*=len(rows)/6545
            if a.preflight_only:
                p.log('[preflight passed] no model loading or GPU generation');return
            stage='load frozen Llama';p.stage(1,1)
            if any(len(v)<len(rows) for v in results.values()):
                if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; --preflight-only is supported')
                torch.manual_seed(42);random.seed(42);torch.backends.cuda.enable_cudnn_sdp(False)
                torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
                model=AutoModelForCausalLM.from_pretrained(cfg.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
                model.generation_config.pad_token_id=tok.pad_token_id
            p.complete('one shared frozen model; unchanged BF16/SDPA')
            rp=a.output_root/'runtime.json';runtime=json.loads(rp.read_text()) if rp.exists() else {}
            for i,arm in enumerate(ARMS,2):
                if runtime.get(arm,{}).get('seconds_per_question'):p.weights[i]=runtime[arm]['seconds_per_question']*len(rows)
            for i,arm in enumerate(ARMS,2):
                stage=arm;p.stage(i,len(rows),initial=len(results[arm]),unit='q',kind='generation')
                pending=sorted((r for r in rows if r['sample_id'] not in results[arm]),key=lambda r:(r['short_lengths'][arm],r['sample_id']))
                is_pced=arm=='pced_cot';streams=9 if is_pced else 1;cap=a.pced_batch_questions if is_pced else a.max_batch_questions
                if pending:
                    engine=QuestionFocusedPCED(model,tok,labels,gamma=ref['contract']['gamma'],max_rationale_tokens=512) if is_pced else QuestionFocusedCoT(model,tok,labels,arm=arm,max_rationale_tokens=512)
                    torch.cuda.reset_peak_memory_stats()
                offset=fresh=0;elapsed=0.
                while offset<len(pending):
                    window=pending[offset:offset+cap];free,total=torch.cuda.mem_get_info()
                    available=max(0,min(free,total*a.memory_fraction-torch.cuda.memory_allocated())-6*1024**3)
                    kv=kv_bytes_per_question(model.config,max(r['short_lengths'][arm] for r in window),528,streams=streams)
                    count=min(len(window),max(1,int(available*.5/max(1,kv))));batch=window[:count]
                    stage=f'{arm} first={batch[0]["sample_id"]} completed={len(results[arm])}/{len(rows)}';start=time.monotonic()
                    try:generated=engine.generate(batch,'rationale') if is_pced else engine.generate_plain(batch,arm)
                    except torch.OutOfMemoryError:
                        gc.collect();torch.cuda.empty_cache()
                        if count==1:raise
                        cap=max(1,count//2);p.log(f'[OOM retry] uncommitted {arm} batch {count}->{cap}; no input/budget changes');continue
                    if {v['sample_id'] for v in generated}!={r['sample_id'] for r in batch} or len(generated)!=len(batch):raise RuntimeError('Incomplete/duplicate batch')
                    expected={r['sample_id']:r for r in batch}
                    for v in generated:
                        v['arm']=arm;v['contract_sha256']=h;validate_record(expected[v['sample_id']],v,arm,h)
                    path=a.output_root/arm/'shards'/('batch_'+hashlib.sha256(batch[0]['sample_id'].encode()).hexdigest()+'.jsonl')
                    if path.exists():raise RuntimeError(f'Refusing overwrite: {path}')
                    atomic_jsonl(path,generated)
                    for v in generated:results[arm][v['sample_id']]=v
                    fresh+=count;offset+=count;elapsed+=time.monotonic()-start;completed[arm]=len(results[arm]);p.update(count)
                    runtime[arm]={'fresh_questions':fresh,'seconds_including_batch_save':elapsed,'seconds_per_question':elapsed/fresh,
                        'last_batch_questions':count,'streams_per_question':streams,'gpu':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                    atomic_json(rp,runtime)
                p.complete(f'{len(results[arm])}/{len(rows)}; {a.output_root/arm}')
            stage='Macro-3 comparison';p.stage(5,1)
            report=matched.summarize(rows,gold,results);report['version']=VERSION
            report['old_prompt_conditions']=matched.summarize(rows,gold,previous)['conditions']
            report['prompt_changes']=prompt_changes(rows,gold,previous,results)
            report['scope']='TEST-informed diagnostic only; direct results not modified; beta is prompt-dependent'
            atomic_json(a.output_root/'summary.json',report)
            atomic_jsonl(a.output_root/'paired_predictions.jsonl',({'sample_id':r['sample_id'],'dataset':r['dataset'],'gold':gold[r['sample_id']],
                'old':{arm:previous[arm][r['sample_id']]['answer'] for arm in ARMS},'new':{arm:results[arm][r['sample_id']]['answer'] for arm in ARMS}} for r in rows))
            p.complete(str(a.output_root/'summary.json'))
            p.log('[complete] Macro-3 '+json.dumps({arm:round(report['conditions'][arm]['macro3_accuracy']*100,3) for arm in ARMS}))
        except Exception as exc:
            p.log(f'[FAILED] stage={stage}; completed={completed}; durable={a.output_root}; {type(exc).__name__}: {exc}; identical command resumes committed shards; historical results unchanged');raise
        finally:p.close()


if __name__=='__main__':run(arguments())

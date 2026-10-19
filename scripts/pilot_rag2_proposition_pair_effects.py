#!/usr/bin/env python3
"""Official-train-only, bounded intervention supervision pilot; no final TEST.

This first pilot does NOT fine-tune Llama or backpropagate the joint QA loss.
It measures local effects, learns them offline, and checks joint deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch.nn import functional as F

from medrag.generation.proposition_kg_attention import PropositionAttentionModel, validate_layout
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, jsonl, sha_file, identity
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.training.proposition_pair_effects import (
    PairEffectController, select_cohort, sampled_pairs, stable_key, supervision_loss,
    training_constants, local_metrics, signal_report, paired_interval,
)

VERSION = 'proposition_pair_effects_v1'
CODE = ['scripts/pilot_rag2_proposition_pair_effects.py',
        'medrag/training/proposition_pair_effects.py',
        'medrag/generation/proposition_kg_attention.py',
        'medrag/training/graph_input_modules.py',
        'medrag/training/graph_input_data.py', 'medrag/training/graph_input_progress.py']
STAGES = ['contracts/cohort', 'load frozen Llama', 'zero/control safety',
          'measure Train128', 'measure Dev32', 'local signal gate',
          'tiny overfit Train16', 'fit effects Train128/Dev32',
          'measure held-out64', 'joint QA held-out64', 'paired report']


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root', type=Path, default=ROOT/'results/rag2_official_train_proposition_attention_v1')
    p.add_argument('--model', type=Path, default=ROOT.parent/'models/Llama-3-8B-Instruct')
    p.add_argument('--output-root', type=Path, default=ROOT/'results/rag2_proposition_pair_effects_pilot_v1')
    p.add_argument('--seed', type=int, default=314159)
    p.add_argument('--train-questions', type=int, default=128)
    p.add_argument('--dev-questions', type=int, default=32)
    p.add_argument('--heldout-questions', type=int, default=64)
    p.add_argument('--pairs-per-question', type=int, default=4)
    p.add_argument('--delta', type=float, default=.5)
    p.add_argument('--effect-floor', type=float, default=.01, help='Minimum useful CE reduction, nats; not a medical threshold')
    p.add_argument('--numerical-ce-tolerance', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--tiny-steps', type=int, default=400)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--batch-questions', type=int, default=8, help='Controller-only training; target Llama comparisons always batch1')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--measure-only', action='store_true', help='Stop after Train/Dev interventions and signal gate; same command resumes later')
    a = p.parse_args(argv)
    if min(a.train_questions,a.dev_questions,a.heldout_questions,a.pairs_per_question,a.epochs,a.tiny_steps,a.batch_questions) <= 0:
        p.error('Counts must be positive')
    if min(a.train_questions,a.dev_questions,a.heldout_questions) < 8:
        p.error('Each split must contain at least eight questions')
    if not 0 < a.delta <= 2 or not 0 < a.numerical_ce_tolerance < a.effect_floor/10 or a.learning_rate <= 0:
        p.error('Invalid delta/numerical floor/learning rate')
    return a


def preflight(a):
    donor = json.loads((a.prepared_root/'manifest.json').read_text())
    complete = json.loads((a.prepared_root/'prepared/COMPLETE.json').read_text())
    rows_path = a.prepared_root/'prepared/rows.jsonl'
    if digest(donor['contract']) != donor['contract_sha256'] or complete['contract_sha256'] != donor['contract_sha256']:
        raise RuntimeError('Prepared donor contract corrupt; do not reuse')
    if sha_file(rows_path) != complete['rows_sha256']:
        raise RuntimeError('Prepared rows checksum mismatch')
    if Path(donor['contract']['model']).resolve() != a.model.resolve() or donor['contract'].get('lora') is not None:
        raise RuntimeError('Require the exact pristine target Llama of the prepared data')
    for item in donor['contract']['inputs']['model_files'].values():
        actual = identity(item['path'], full_hash='sha256' in item)
        if any(actual.get(k) != v for k,v in item.items()):
            raise RuntimeError(f'Model/tokenizer identity changed: {item["path"]}')
    rows = list(jsonl(rows_path))
    if Counter(r['split'] for r in rows) != {'train':9000,'validation':1000}:
        raise RuntimeError('Require official Train9000/Val1000, not the old TEST re-split')
    cohort = select_cohort(rows,a.seed,a.train_questions,a.dev_questions,a.heldout_questions)
    for r in cohort:
        validate_layout(r)
        if len(r['input_ids']) > 8064 or len(r['choice_token_ids']) != 4 or len(set(r['choice_token_ids'])) != 4:
            raise RuntimeError('Unexpected context or answer-token layout')
    contract = dict(version=VERSION, hypothesis='Measured pair-specific effects teach useful input-conditioned coupling beyond layer constants.',
        source_manifest=donor, prepared_rows_sha256=complete['rows_sha256'], model=str(a.model.resolve()),
        code={p:sha_file(ROOT/p) for p in CODE}, environment={'torch':torch.__version__,
            'transformers':__import__('transformers').__version__},
        cohort=[dict(sample_id=r['sample_id'],dataset=r['dataset'],split=r['pilot_split'],
                     row_sha256=digest(r),selected_pairs=sampled_pairs(r,a.pairs_per_question,a.seed)) for r in cohort],
        settings={k:getattr(a,k) for k in ('seed','train_questions','dev_questions','heldout_questions',
                   'pairs_per_question','delta','effect_floor','numerical_ce_tolerance','epochs','tiny_steps',
                   'learning_rate','batch_questions')},
        fixed={'controller_hidden':128,'graph_rounds':2,'feature_cut':12,'layers':[12,13,14,15],
            'head_sharing':True,'target_batch':1,'dtype':'BF16 frozen Llama; FP32 controller and selected lm_head',
            'backend':'same enabled block-mask path for zero/intervention/constant/learned; no cuDNN',
            'candidate_rule':'existing KG shared endpoint pairs, causal receiver > sender',
            'intervention':'one pair and ONE layer at a time; other biases exactly zero; all 4 layers per sampled pair',
            'loss':'question-balanced SmoothL1(predicted CE gains / train scale, measured gains / scale); weak gains zeroed',
            'effect_scale':'max(.02, train-only 90th percentile absolute measured CE effect)',
            'optimizer':'AdamW lr specified; weight_decay .01; clip norm 1; deterministic epoch shuffle',
            'selection':'min Dev local action regret, earlier epoch on ties; heldout remains unseen',
            'local_metric':'noise-floor-adjusted action regret, equal weight per question; raw effects also saved',
            'inference':'max predicted gain > effect_floor -> signed delta; otherwise zero; no labels/options as features',
            'graphless':'retained, zero action and zero local regret',
            'tiny_gate':'Train16 informative action accuracy >=90% and regret <=10% of zero baseline',
            'signal_gate':'train >=32 informative decisions in >=16 questions AND >=16 deviations from layer-constant actions',
            'heldout_gate':'regret >=20% lower than layer constants and paired question-bootstrap95% lower bound >0; joint CE at least .01 lower than zero AND constant; correct count >= both',
            'scope':'local labels condition on other biases zero; joint deployment is a separate check; no rationale/free-generation claim; no final TEST',
            'limits':'sampling four existing causal pairs does not prove all-pair or same-pair/across-question generalization',
            'bootstrap':'2000 paired question resamples, fixed seed', 'llm_training':False,'joint_qa_finetuning':False})
    h = digest(contract)
    path = a.output_root/'manifest.json'
    if path.exists() and json.loads(path.read_text()).get('contract_sha256') != h:
        raise RuntimeError(f'Pilot resume contract mismatch. Preserve existing data; use a new --output-root: {a.output_root}')
    ancestor = a.output_root
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free < 3*1024**3:
        raise RuntimeError('Need at least 3 GiB free for bounded pilot caches')
    return cohort, contract, h


def save_torch(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as handle:
        torch.save(data,handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp,path)


def read_cache(path, h, row=None):
    value = torch.load(path,map_location='cpu',weights_only=False)
    if value['contract_sha256'] != h or (row is not None and value['row_sha256'] != digest(row)):
        raise RuntimeError(f'Incompatible cache: {path}')
    return value


def cache_path(a, row):
    return a.output_root/'measurements'/row['pilot_split']/(stable_key(0,row['sample_id'])+'.pt')


@torch.no_grad()
def target_score(wrapper, row, pad_id):
    with training_attention_backend():
        out=wrapper([row],pad_id=pad_id)
    ids=torch.tensor(row['choice_token_ids'],device=out.last_hidden.device)
    logits=F.linear(out.last_hidden.float(),wrapper.llm.lm_head.weight.index_select(0,ids).float())[0]
    gold=int(row['gold_index'])
    ce=F.cross_entropy(logits[None],torch.tensor([gold],device=logits.device))
    if not torch.isfinite(logits).all() or not torch.isfinite(ce):
        raise RuntimeError('Nonfinite target result')
    return dict(logits=logits.tolist(),ce=float(ce),p_gold=float(logits.softmax(-1)[gold]),
                prediction=int(logits.argmax()),gold=gold,correct=bool(logits.argmax().item()==gold))


def measure_one(a, wrapper, row, pad_id, h):
    path=cache_path(a,row)
    if path.exists(): return read_cache(path,h,row)
    c=wrapper.controller
    c.mode='zero'; c.capture=True
    base=target_score(wrapper,row,pad_id)
    features=c.captured; c.capture=False; c.captured=None
    selected=sampled_pairs(row,a.pairs_per_question,a.seed)
    measurements=[]
    for index in selected:
        layer_effects,raw=[],[]
        for layer in range(c.layers):
            scores=[]
            for direction in (-1,1):
                c.mode='intervention'; c.intervention=(index,layer,direction)
                scores.append(target_score(wrapper,row,pad_id))
            raw.append(scores); layer_effects.append([base['ce']-v['ce'] for v in scores])
        receiver,sender=row['pairs'][index]
        measurements.append(dict(pair_index=index,receiver=row['proposition_ids'][receiver],
                                 sender=row['proposition_ids'][sender],effects=layer_effects,raw=raw))
    c.mode='zero'; c.intervention=None
    repeated=target_score(wrapper,row,pad_id)
    noise=abs(base['ce']-repeated['ce'])
    if noise>a.numerical_ce_tolerance or base['prediction']!=repeated['prediction']:
        raise RuntimeError(f'Zero-control repeat unstable: {row["sample_id"]}, CE delta={noise}; no labels committed')
    record=dict(contract_sha256=h,row_sha256=digest(row),sample_id=row['sample_id'],
        dataset=row['dataset'],pilot_split=row['pilot_split'],features=features,layers=c.layers,
        selected_pairs=selected,base=base,zero_repeat=repeated,numerical_ce_difference=noise,measurements=measurements)
    save_torch(path,record)
    return record


def measure_split(a, wrapper, rows, pad_id, h, progress, stage):
    cached=sum(cache_path(a,r).exists() for r in rows)
    progress.stage(stage,len(rows),initial=cached,unit='q',kind='measurement')
    done={r['sample_id']:read_cache(cache_path(a,r),h,r) for r in rows if cache_path(a,r).exists()}
    fresh=0
    for r in rows:
        if r['sample_id'] in done: continue
        done[r['sample_id']]=measure_one(a,wrapper,r,pad_id,h)
        fresh+=1; progress.update()
        if fresh>=4:
            rate=(time.monotonic()-progress.stage_started)/fresh
            for index,part in [(3,'train'),(4,'dev'),(8,'heldout')]:
                if index>=stage:
                    n={'train':a.train_questions,'dev':a.dev_questions,'heldout':a.heldout_questions}[part]
                    progress.weights[index]=rate*n
            progress.weights[9]=rate*a.heldout_questions*3/(2+8*a.pairs_per_question)
    progress.complete(f'cached={cached}; new={fresh}; {a.output_root/"measurements"}')
    return [done[r['sample_id']] for r in rows]


def cpu_state(c):
    return {k:v.detach().cpu().clone() for k,v in c.state_dict().items()}


def new_controller(a, dim, device, constants, scale):
    torch.manual_seed(a.seed)
    c=PairEffectController(dim,delta=a.delta,floor=a.effect_floor).to(device)
    c.constant_effects.copy_(torch.tensor(constants,device=device,dtype=torch.float32))
    c.effect_scale.fill_(scale)
    return c


def update_controller(c, optimizer, records):
    optimizer.zero_grad(set_to_none=True)
    losses=[supervision_loss(c,r) for r in records]
    losses=[v for v in losses if v is not None]
    if not losses: return 0.
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite effect-regression loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(c.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def tiny_fit(a, dim, device, records, constants, scale, h, progress):
    path=a.output_root/'tiny/COMPLETE.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['contract_sha256']!=h: raise RuntimeError('Tiny-fit contract mismatch')
        progress.stage(6,1,initial=1); progress.complete('cached tiny overfit'); return result['passed']
    # Deliberately informative TRAIN examples for an overfit test, never heldout.
    eligible=[r for r in records if any(max(max(v) for v in m['effects'])>a.effect_floor for m in r['measurements'])]
    sample=sorted(eligible,key=lambda r:stable_key(a.seed,'tiny',r['sample_id']))[:16]
    if len(sample)<16: raise RuntimeError('Signal gate permitted fewer than 16 informative train questions')
    c=new_controller(a,dim,device,constants,scale)
    opt=torch.optim.AdamW(c.parameters(),lr=a.learning_rate,weight_decay=.01)
    progress.stage(6,a.tiny_steps,unit='step',kind='fit')
    initial,_=local_metrics(c,sample); metrics=initial
    for step in range(a.tiny_steps):
        rng=random.Random(a.seed+step); batch=rng.sample(sample,min(a.batch_questions,len(sample)))
        update_controller(c,opt,batch); progress.update()
        if (step+1)%25==0 or step+1==a.tiny_steps:
            metrics,_=local_metrics(c,sample)
            if metrics['informative_action_accuracy']>=.9 and metrics['learned_regret']<=.1*metrics['zero_regret']:
                break
    passed=metrics['informative_action_accuracy']>=.9 and metrics['learned_regret']<=.1*metrics['zero_regret']
    atomic_json(path,dict(contract_sha256=h,passed=passed,steps=step+1,
                         sample_ids=[r['sample_id'] for r in sample],initial=initial,final=metrics))
    progress.complete(f'passed={passed}; {path}')
    return passed


def fit(a, dim, device, train, dev, constants, scale, h, progress):
    c=new_controller(a,dim,device,constants,scale)
    opt=torch.optim.AdamW(c.parameters(),lr=a.learning_rate,weight_decay=.01)
    root=a.output_root/'controller'; last=root/'last.pt'; best_path=root/'best.pt'
    epoch=0; history=[]; best_metric=float('inf'); best_state=cpu_state(c)
    if last.exists():
        saved=read_cache(last,h); c.load_state_dict(saved['controller']); opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor): state[key]=value.to(device)
        epoch=saved['epoch']; history=saved['history']; best_metric=saved['best_metric']
        best_state=read_cache(best_path,h)['controller']
    else:
        initial,_=local_metrics(c,dev)
        best_metric=initial['learned_regret']
        save_torch(best_path,dict(contract_sha256=h,epoch=0,controller=best_state,dev=initial))
    progress.stage(7,a.epochs*len(train),initial=epoch*len(train),unit='q',kind='fit')
    while epoch<a.epochs:
        progress.names[7]=f'effect train epoch{epoch+1}/{a.epochs}'
        order=list(range(len(train))); random.Random(a.seed+epoch).shuffle(order)
        losses=[]
        for start in range(0,len(order),a.batch_questions):
            batch=[train[i] for i in order[start:start+a.batch_questions]]
            losses.append(update_controller(c,opt,batch)); progress.update(len(batch))
        progress.names[7]=f'effect dev epoch{epoch+1}/{a.epochs}'
        metric,per_question=local_metrics(c,dev)
        train_metric,_=local_metrics(c,train)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),train=train_metric,dev=metric))
        if metric['learned_regret']<best_metric:
            best_metric=metric['learned_regret']; best_state=cpu_state(c)
            save_torch(best_path,dict(contract_sha256=h,epoch=epoch+1,controller=best_state,dev=metric))
        epoch+=1
        save_torch(last,dict(contract_sha256=h,epoch=epoch,controller=cpu_state(c),optimizer=opt.state_dict(),
                            history=history,best_metric=best_metric))
        atomic_json(root/'history.json',history)
    c.load_state_dict(best_state); c.eval()
    progress.complete(f'best Dev regret={best_metric:.6f}; {best_path}')
    return c


def qa_summary(records):
    counts=Counter(r['dataset'] for r in records)
    correct=Counter(r['dataset'] for r in records if r['score']['correct'])
    return dict(n=len(records),counts=dict(counts),correct=sum(correct.values()),
        micro=sum(correct.values())/len(records),macro2=float(np.mean([correct[d]/n for d,n in counts.items()])),
        ce=float(np.mean([r['score']['ce'] for r in records])),
        p_gold=float(np.mean([r['score']['p_gold'] for r in records])),missing=0)


def joint_eval(a, wrapper, rows, pad_id, h, progress):
    out=[]; cached=0
    best_hash=sha_file(a.output_root/'controller/best.pt')
    jobs=[(row,mode) for row in rows for mode in ('zero','constant','learned')]
    def path(row,mode): return a.output_root/'joint'/mode/(stable_key(0,row['sample_id'])+'.json')
    cached=sum(path(r,m).exists() for r,m in jobs)
    progress.stage(9,len(jobs),initial=cached,unit='forward',kind='validation')
    for row,mode in jobs:
        p=path(row,mode)
        if p.exists():
            record=json.loads(p.read_text())
            if record['contract_sha256']!=h or record['checkpoint_sha256']!=best_hash:
                raise RuntimeError('Joint checkpoint/cache mismatch')
        else:
            wrapper.controller.mode=mode
            record=dict(contract_sha256=h,checkpoint_sha256=best_hash,sample_id=row['sample_id'],
                        dataset=row['dataset'],condition=mode,score=target_score(wrapper,row,pad_id))
            atomic_json(p,record); progress.update()
        out.append(record)
    atomic_jsonl(a.output_root/'joint_predictions.jsonl',out)
    progress.complete(f'{len(jobs)} conditions, {len(rows)} heldout questions; {a.output_root/"joint_predictions.jsonl"}')
    return out


def report(a, controller, records, joint, signal, h):
    local,per_question=local_metrics(controller,records)
    deltas=[r['constant_regret']-r['learned_regret'] for r in per_question]
    interval=paired_interval(deltas,a.seed)
    reduction=(local['constant_regret']-local['learned_regret'])/max(1e-12,local['constant_regret'])
    groups={mode:[r for r in joint if r['condition']==mode] for mode in ('zero','constant','learned')}
    qa={mode:qa_summary(rows) for mode,rows in groups.items()}
    transitions={}
    paired=[]
    for mode in ('zero','constant'):
        base={r['sample_id']:r['score'] for r in groups[mode]}
        ds=[]; change=Counter()
        for row in groups['learned']:
            b=base[row['sample_id']]; c=row['score']; d=b['ce']-c['ce']; ds.append(d)
            key=('C' if b['correct'] else 'W')+'->'+('C' if c['correct'] else 'W'); change[key]+=1
            paired.append(dict(sample_id=row['sample_id'],baseline=mode,transition=key,ce_improvement=d,
                               p_gold_delta=c['p_gold']-b['p_gold']))
        transitions[mode]=dict(counts=dict(change),ce_improvement=float(np.mean(ds)),
                               ce_improvement_ci95=paired_interval(ds,a.seed))
    local_pass=local['constant_regret']>0 and reduction>=.2 and interval[0]>0
    joint_pass=all(qa['learned']['ce']<=qa[m]['ce']-.01 and qa['learned']['correct']>=qa[m]['correct'] for m in ('zero','constant'))
    result=dict(contract_sha256=h,status='complete',signal=signal,local=local,
                local_regret_reduction_vs_constant=reduction,local_gain_vs_constant_ci95=interval,
                local_pass=local_pass,joint=qa,paired=transitions,joint_pass=joint_pass,
                pilot_pass=bool(local_pass and joint_pass),
                scope='Official-train development only; target model-specific MCQ selection, not final TEST or free generation.',
                next='Independent larger validation replication, not final TEST tuning.' if local_pass and joint_pass else
                     'Do not scale. Inspect local labels vs controller fit vs joint interaction separately.')
    atomic_jsonl(a.output_root/'local_heldout.jsonl',per_question)
    atomic_jsonl(a.output_root/'paired_heldout.jsonl',paired)
    return result


def run(a, cohort, contract, h, progress):
    groups={part:[r for r in cohort if r['pilot_split']==part] for part in ('train','dev','heldout')}
    progress.names[3]=f'measure Train{len(groups["train"])}'
    progress.names[4]=f'measure Dev{len(groups["dev"])}'
    progress.names[8]=f'measure held-out{len(groups["heldout"])}'
    progress.weights=[1,3,8,None,None,1,None,None,None,None,1]
    progress.stage(0,1)
    mp=a.output_root/'manifest.json'
    if not mp.exists():
        provenance={}
        for key,args in [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current'])]:
            provenance[key]=subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
        atomic_json(mp,dict(contract=contract,contract_sha256=h,provenance=provenance))
    atomic_jsonl(a.output_root/'cohort.jsonl',contract['cohort'])
    progress.complete(f'official Train/Val only; no TEST; {mp}')
    if (a.output_root/'SUMMARY.json').exists():
        result=json.loads((a.output_root/'SUMMARY.json').read_text())
        if result['contract_sha256']!=h: raise RuntimeError('Summary contract mismatch')
        return result
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available():
        raise RuntimeError('Use the launcher with CUDA_VISIBLE_DEVICES=1 and functioning GPU driver')
    progress.stage(1,1)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error(); logging.disable_progress_bar()
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    llm=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,
                                           device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    torch.manual_seed(a.seed)
    c=PairEffectController(llm.config.hidden_size,delta=a.delta,floor=a.effect_floor).to('cuda')
    wrapper=PropositionAttentionModel(llm,c,gradient_checkpointing=False).eval()
    progress.complete('pristine frozen Llama; new utility controller, no old controller/LoRA loaded')
    progress.stage(2,1)
    sample=next(r for r in groups['train'] if r['pairs'])
    c.mode='zero'; first=target_score(wrapper,sample,pad); repeated=target_score(wrapper,sample,pad)
    if abs(first['ce']-repeated['ce'])>a.numerical_ce_tolerance:
        raise RuntimeError('Repeated zero-bias CE unstable')
    if any(p.requires_grad for p in llm.parameters()): raise RuntimeError('Target is not frozen')
    c.capture=True; target_score(wrapper,sample,pad); original=c.captured; c.capture=False
    altered=dict(sample); altered['input_ids']=list(sample['input_ids'])
    end=sample['question_span'][1]
    altered['input_ids'][end:]=[pad]*len(altered['input_ids'][end:])
    altered['gold_index']=(sample['gold_index']+1)%4
    c.capture=True; target_score(wrapper,altered,pad); modified=c.captured; c.capture=False; c.captured=None
    for key in ('q','p','pairs','roles'): torch.testing.assert_close(original[key],modified[key],atol=0,rtol=0)
    atomic_json(a.output_root/'SAFETY.json',dict(contract_sha256=h,zero_repeat_ce_delta=abs(first['ce']-repeated['ce']),
                target_frozen=True,choice_head_fp32=True,answer_suffix_and_gold_excluded=True,gpu=torch.cuda.get_device_name(0)))
    progress.complete('same enabled mask path; FP32 choice head; no choice/gold in controller features')
    train=measure_split(a,wrapper,groups['train'],pad,h,progress,3)
    dev=measure_split(a,wrapper,groups['dev'],pad,h,progress,4)
    progress.stage(5,1)
    signal=signal_report(train,a.effect_floor)
    passed=signal['informative_decisions']>=32 and signal['informative_questions']>=16 and signal['differs_from_layer_constant']>=16
    signal['passed']=bool(passed)
    atomic_json(a.output_root/'SIGNAL.json',signal)
    audit=[]
    for record in train+dev:
        audit.append({k:v for k,v in record.items() if k!='features'})
    atomic_jsonl(a.output_root/'train_dev_interventions.jsonl',audit)
    progress.complete(f'passed={passed}; {a.output_root/"SIGNAL.json"}')
    if not passed: return dict(contract_sha256=h,status='stopped_signal_gate',signal=signal,pilot_pass=False)
    if a.measure_only: return dict(contract_sha256=h,status='measurement_only',signal=signal)
    constants,scale=training_constants(train,floor=a.effect_floor)
    if not tiny_fit(a,llm.config.hidden_size,llm.device,train,constants,scale,h,progress):
        return dict(contract_sha256=h,status='stopped_tiny_overfit',signal=signal,pilot_pass=False)
    learned=fit(a,llm.config.hidden_size,llm.device,train,dev,constants,scale,h,progress)
    # Heldout labels first measured AFTER checkpoint selection is frozen.
    wrapper.controller=c
    heldout=measure_split(a,wrapper,groups['heldout'],pad,h,progress,8)
    atomic_jsonl(a.output_root/'heldout_interventions.jsonl',
                 [{k:v for k,v in r.items() if k!='features'} for r in heldout])
    wrapper.controller=learned
    joint=joint_eval(a,wrapper,groups['heldout'],pad,h,progress,9)
    progress.stage(10,1)
    result=report(a,learned,heldout,joint,signal,h)
    progress.complete(f'local_pass={result["local_pass"]}; joint_pass={result["joint_pass"]}; {a.output_root/"SUMMARY.json"}')
    return result


def main(argv=None):
    a=arguments(argv)
    torch.set_num_threads(8)
    cohort,contract,h=preflight(a)
    counts=Counter((r['pilot_split'],r['dataset']) for r in cohort)
    forwards=sum(2+8*len(sampled_pairs(r,a.pairs_per_question,a.seed)) for r in cohort)
    if a.preflight_only:
        print(json.dumps(dict(status='preflight_ok',contract_sha256=h,
            counts={f'{p}/{d}':n for (p,d),n in counts.items()},measurement_forwards=forwards,
            cached_questions=sum(cache_path(a,r).exists() for r in cohort),max_prompt_tokens=max(len(r['input_ids']) for r in cohort),
            output=str(a.output_root),no_gpu_loaded=True,no_experiment_started=True),indent=2)); return
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=Progress(list(STAGES))
        try:
            cached=sum(cache_path(a,r).exists() for r in cohort)
            progress.log(f'[plan] contracts -> frozen Llama -> safety -> Train/Dev interventions -> signal gate -> Train16 overfit -> controller fit -> untouched heldout interventions -> 3-way joint QA -> report; cached={cached}/{len(cohort)} questions; remaining={len(cohort)-cached}; measurement <= {forwards} forwards; no final TEST')
            progress.log('[ETA] approximately 15-40 minutes on idle H200 GPU1; batch1 same-path comparisons; calibration after 4 fresh questions; stop on failed gates; per-question/epoch resume')
            result=run(a,cohort,contract,h,progress)
            if result['status']!='measurement_only': atomic_json(a.output_root/'SUMMARY.json',result)
            else: atomic_json(a.output_root/'MEASUREMENT_ONLY.json',result)
            for i in range(progress.index+1,len(progress.names)):
                progress.stage(i,1,initial=1); progress.complete(f'not executed: {result["status"]}')
            progress.log(f'[result] status={result["status"]}; pilot_pass={result.get("pilot_pass", "not evaluated")}; {a.output_root}')
        except Exception as error:
            progress.log(f'[FAILED] stage={progress.index+1}/{len(progress.names)} item={progress.done}/{progress.total}; {type(error).__name__}: {error}; durable={a.output_root}; identical command resumes completed questions/epochs; no old results deleted')
            raise
        finally: progress.close()


if __name__=='__main__': main()

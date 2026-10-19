#!/usr/bin/env python3
"""A: controllability; B: effect prediction; C: joint deployment, official TRAIN only.

All numerical actions, cohorts, gates and baselines are frozen before execution.
This diagnoses a finite intervention family, not impossibility of KG reasoning.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from medrag.generation.proposition_kg_attention import PropositionAttentionModel, validate_layout
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, identity, jsonl, sha_file
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.training.proposition_pair_effects import stable_key, paired_interval
from medrag.training.attention_intervention_diagnostic import (
    GROUPS, DOSES, ACTIONS, cohort, candidates, candidate_row, reverse_propositions, subset,
    InterventionController, AttentionProbe, install_probe, features, EffectPredictor,
    gains, select_actions, chosen_gain, compare_scores,
)

VERSION = 'attention_identifiability_v1'
CODE = ['scripts/pilot_rag2_attention_identifiability.py',
        'medrag/training/attention_intervention_diagnostic.py',
        'medrag/generation/proposition_kg_attention.py', 'medrag/training/proposition_pair_effects.py',
        'medrag/training/graph_input_modules.py', 'medrag/training/graph_input_data.py',
        'medrag/training/graph_input_progress.py']
STAGES = ['contract/cohorts', 'load frozen Llama', 'numerical/feature safety',
          'A: measure capability', 'A: reverse order', 'A: headroom gate',
          'B: measure train', 'B: measure dev', 'B: tiny overfit', 'B: fit predictor',
          'C: heldout interventions', 'C: joint application', 'paired diagnosis']


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',type=Path,default=ROOT/'results/rag2_official_train_proposition_attention_v1')
    p.add_argument('--exclude-cohort',type=Path,default=ROOT/'results/rag2_proposition_pair_effects_pilot_v1/cohort.jsonl')
    p.add_argument('--model',type=Path,default=ROOT.parent/'models/Llama-3-8B-Instruct')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/rag2_attention_identifiability_v1')
    p.add_argument('--phase',choices=['all','capability'],default='all')
    p.add_argument('--seed',type=int,default=271828)
    p.add_argument('--capability-questions',type=int,default=128)
    p.add_argument('--train-questions',type=int,default=128)
    p.add_argument('--dev-questions',type=int,default=32)
    p.add_argument('--heldout-questions',type=int,default=64)
    p.add_argument('--order-questions',type=int,default=16)
    p.add_argument('--pairs-per-question',type=int,default=4)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--tiny-steps',type=int,default=400)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--effect-floor',type=float,default=.01)
    p.add_argument('--noise-tolerance',type=float,default=1e-4)
    p.add_argument('--preflight-only',action='store_true')
    a = p.parse_args(argv)
    if min(a.capability_questions,a.train_questions,a.dev_questions,a.heldout_questions) < 8:
        p.error('Each cohort must have at least 8 questions')
    if min(a.pairs_per_question,a.epochs,a.tiny_steps,a.learning_rate) <= 0:
        p.error('Counts and learning rate must be positive')
    if not 0 <= a.order_questions <= a.capability_questions or not 0 < a.noise_tolerance < a.effect_floor/10:
        p.error('Invalid order count or numerical floor')
    return a


def preflight(a):
    donor = json.loads((a.prepared_root/'manifest.json').read_text())
    complete = json.loads((a.prepared_root/'prepared/COMPLETE.json').read_text())
    path = a.prepared_root/'prepared/rows.jsonl'
    if digest(donor['contract']) != donor['contract_sha256'] or complete['contract_sha256'] != donor['contract_sha256']:
        raise RuntimeError('Prepared source manifest is inconsistent')
    if sha_file(path) != complete['rows_sha256']:
        raise RuntimeError('Prepared row checksum changed')
    if Path(donor['contract']['model']).resolve() != a.model.resolve() or donor['contract'].get('lora') is not None:
        raise RuntimeError('Require the prepared pristine Llama, without old LoRA')
    for value in donor['contract']['inputs']['model_files'].values():
        actual = identity(value['path'],full_hash='sha256' in value)
        if any(actual.get(k) != v for k,v in value.items()):
            raise RuntimeError(f'Model/tokenizer file changed: {value["path"]}')
    rows = list(jsonl(path))
    if Counter(r['split'] for r in rows) != {'train':9000,'validation':1000}:
        raise RuntimeError('Require official Train9000/Val1000 cache')
    excluded = {r['sample_id'] for r in jsonl(a.exclude_cohort)}
    sizes = {s:getattr(a,s+'_questions') for s in ('capability','train','dev','heldout')}
    selected = cohort(rows,excluded,a.seed,sizes)
    plans = {}
    for row in selected:
        validate_layout(row)
        if len(row['choice_token_ids']) != 4 or len(set(row['choice_token_ids'])) != 4:
            raise RuntimeError('Expected four distinct, single-token answer labels')
        plans[row['sample_id']] = candidates(row,a.pairs_per_question,a.seed)
    contract = dict(version=VERSION, source_manifest=donor, model=str(a.model.resolve()),
        prepared_sha256=complete['rows_sha256'],excluded_cohort_sha256=sha_file(a.exclude_cohort),
        settings={k:v for k,v in vars(a).items() if k not in
                  ('prepared_root','exclude_cohort','model','output_root','phase','preflight_only')},
        cohort=[dict(sample_id=r['sample_id'],dataset=r['dataset'],split=r['diagnostic_split'],
                     row_sha256=digest(r),candidates=plans[r['sample_id']]) for r in selected],
        code={p:sha_file(ROOT/p) for p in CODE},
        environment=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),
        fixed=dict(hypothesis='Useful conditional pair effects exist, can be predicted, and survive joint application.',
          scope='Official TRAIN only; original Val1000 and final TEST6545 untouched; frozen Llama MCQ selection.',
          groups_one_based=[[i+1 for i in g] for g in GROUPS],doses=list(DOSES),
          schedule='All 12 layers use identical enabled masks for zero and every action. Four-layer group is perturbed jointly.',
          inputs='Original exact proposition tokens, question stem, original Top8 and KG shared-endpoint pairs.',
          candidates='Hash-sampled causal graph pairs; equal-count nonedges matched by length, distance, rank, token overlap, document identity.',
          order='Reverse proposition body token blocks; same token multiset, labels/wrappers/options; compare against own zero. Direction also reverses.',
          attention_probe='Up to eight receiver query tokens, all heads, first layer of each group; actual sender-span softmax mass.',
          precision='BF16 target; FP32 selected output head, effects and predictor; no cuDNN; target batch1.',
          features='Frozen layer12 mean+last states of question/propositions; original-KG neighbor means; whole-set mean; covariates+endpoint roles. No options or gold.',
          predictor='Shared LayerNorm/linear 8192->32; concatenated six roles +12 covariates; MLP64->18 CE effects.',
          loss='Question-balanced SmoothL1 of measured CE gains/train-only scale; abs(gain)>floor gets weight2. Weak cases retained.',
          no_op='Predict max gain <=floor: no correction.',
          capability_gate='Restricted gold oracle available-correct gain >=4pp AND best-CE reduction >=.01. Diagnostic oracle is never an inference policy.',
          tiny_gate='At least 8 informative train questions; recover >=80% train16 oracle CE gain.',
          selection='Minimum Dev graph top1 CE regret; earlier epoch on ties; initial zero model eligible.',
          heldout_gate='Graph top1 CE beats zero and train-constant by >.01 with paired95 lower>0; accuracy +2pp vs zero. Small-pilot screen only.',
          controls='Zero, train-global action, uniform random/no-op, graph predicted top1/top2, matched non-graph predicted top1.',
          causal_limits='Intervention effects are target/position/layer/choice-loss specific, not semantic correctness or unrestricted graph impossibility.',
          scaling='No automatic 512/2000/full training. Change train count only in a new versioned output after pilot review.'))
    h = digest(contract)
    mp = a.output_root/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()).get('contract_sha256') != h:
        raise RuntimeError(f'Resume contract mismatch; preserve this run and choose a new output root: {a.output_root}')
    parent = a.output_root
    while not parent.exists(): parent=parent.parent
    if shutil.disk_usage(parent).free < 4*1024**3:
        raise RuntimeError('Need >=4 GiB free for pilot feature caches')
    return selected, plans, contract, h


def save_tensor(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def read_tensor(path,h):
    r = torch.load(path,map_location='cpu',weights_only=False)
    if r['contract_sha256'] != h:
        raise RuntimeError(f'Cache contract mismatch: {path}')
    return r


def cache_path(a,row,order='original'):
    return a.output_root/'measurements'/row['diagnostic_split']/order/(stable_key(0,row['sample_id'])+'.pt')


@torch.no_grad()
def score(w,row,pad):
    with training_attention_backend():
        hidden = w([row],pad_id=pad).last_hidden
    ids = torch.tensor(row['choice_token_ids'],device=hidden.device)
    logits = F.linear(hidden.float(),w.llm.lm_head.weight[ids].float())[0]
    gold = int(row['gold_index'])
    ce = F.cross_entropy(logits[None],torch.tensor([gold],device=hidden.device))
    if not torch.isfinite(logits).all() or not torch.isfinite(ce):
        raise RuntimeError('Nonfinite target score')
    others = [i for i in range(len(ids)) if i != gold]
    return dict(logits=logits.tolist(),ce=float(ce),p_gold=float(logits.softmax(-1)[gold]),
                margin=float(logits[gold]-logits[others].max()),prediction=int(logits.argmax()),
                gold=gold,correct=bool(int(logits.argmax())==gold))


def set_actions(w,actions):
    w.controller.actions = [(pair,g*4+l,dose) for pair,g,dose in actions for l in range(4)]


def measure_one(a,w,probe,row,items,pad,h,order='original',capture_mass=False):
    path = cache_path(a,row,order)
    if path.exists():
        old = read_tensor(path,h)
        if old['row_sha256'] != digest(row) or old['items'] != items:
            raise RuntimeError('Measurement row/candidate mismatch')
        return old
    model_row = candidate_row(row,items)
    set_actions(w,[])
    probe.request = None
    w.controller.capture = True
    base = score(w,model_row,pad)
    f = features(w.controller.captured,row,items)
    w.controller.capture = False; w.controller.captured = None
    masses = []
    if capture_mass and items:
        r,s = items[0]['pair']
        for group in GROUPS:
            probe.request = dict(layer=group[0],receiver_span=row['proposition_spans'][r],sender_span=row['proposition_spans'][s])
            probe.values=[]; check=score(w,model_row,pad)
            if abs(check['ce']-base['ce']) > a.noise_tolerance:
                raise RuntimeError('Observational attention probe changed output')
            masses.append(probe.values[0])
    probe.request=None
    scores, probes = [], []
    for pair,item in enumerate(items):
        grouped=[]
        for g in range(len(GROUPS)):
            values=[]
            for dose in DOSES:
                set_actions(w,[(pair,g,dose)])
                if capture_mass and pair==0 and abs(dose)==2:
                    r,s=item['pair']
                    probe.request=dict(layer=GROUPS[g][0],receiver_span=row['proposition_spans'][r],sender_span=row['proposition_spans'][s])
                    probe.values=[]
                values.append(score(w,model_row,pad))
                if probe.request is not None:
                    probes.append(dict(group=g,dose=dose,zero=masses[g],changed=probe.values[0]))
                probe.request=None
            grouped.append(values)
        scores.append(grouped)
    set_actions(w,[])
    repeated=score(w,model_row,pad)
    noise=abs(base['ce']-repeated['ce'])
    if noise>a.noise_tolerance or base['prediction']!=repeated['prediction']:
        raise RuntimeError(f'Zero repeat unstable: {row["sample_id"]}, CE delta={noise}')
    record=dict(contract_sha256=h,row_sha256=digest(row),sample_id=row['sample_id'],
                dataset=row['dataset'],split=row['diagnostic_split'],order=order,items=items,
                base=base,baselines=[base]*len(GROUPS),scores=scores,features=f,
                zero_repeat=repeated,noise=noise,attention_mass=probes)
    save_tensor(path,record)
    return record


def measure(a,w,probe,rows,plans,pad,h,progress,index,order='original'):
    cached=sum(cache_path(a,r,order).exists() for r in rows)
    progress.stage(index,len(rows),initial=cached,unit='q',kind='measure')
    out=[]; fresh=0
    probe_ids={r['sample_id'] for r in subset(rows,a.order_questions,a.seed)} if index==3 else set()
    for i,row in enumerate(rows):
        items=plans[row['sample_id']]
        if order=='reversed': row,items=reverse_propositions(row,items)
        exists=cache_path(a,row,order).exists()
        out.append(measure_one(a,w,probe,row,items,pad,h,order,
                               capture_mass=row['sample_id'] in probe_ids))
        if not exists:
            progress.update(); fresh+=1
            if fresh>=4:
                per_q=(time.monotonic()-progress.stage_started)/fresh
                for j,n in ((4,a.order_questions),(6,a.train_questions),(7,a.dev_questions),(10,a.heldout_questions)):
                    if index<j<len(progress.names):progress.weights[j]=per_q*n
                if len(progress.names)>11:
                    progress.weights[11]=per_q*a.heldout_questions*6/(2+2*a.pairs_per_question*len(ACTIONS))
    progress.complete(f'n={len(rows)}, cached={cached}, new={len(rows)-cached}; {a.output_root/"measurements"}')
    return out


def audit_record(r):
    return {k:v for k,v in r.items() if k!='features'}


def capability_report(a,records,reversed_records,h):
    baselines=[r['base'] for r in records]; datasets=[r['dataset'] for r in records]
    result=dict(contract_sha256=h,n=len(records),missing=0,
                graphless_or_unmatched=sum(not r['items'] for r in records),
                max_repeat_ce_difference=max(r['noise'] for r in records),
                scope='Gold diagnostic oracle over sampled pairs and 18 actions; not a deployable model.')
    for kind in ('graph','non_graph'):
        bests=[]; available=0; curves=[]
        for r in records:
            outcomes=[s for i,groups in enumerate(r['scores']) if r['items'][i]['kind']==kind
                      for group in groups for s in group]
            bests.append(min([r['base']]+outcomes,key=lambda s:s['ce']))
            available+=int(any(s['correct'] for s in [r['base']]+outcomes))
        report=compare_scores(baselines,bests,datasets)
        report['available_correct_oracle']=available
        report['available_accuracy_delta_pp']=100*(available-sum(b['correct'] for b in baselines))/len(records)
        for g in range(len(GROUPS)):
            for magnitude in (.5,1.,2.):
                delta=[]
                for r in records:
                    vals=[r['base']['ce']-s['ce'] for i,v in enumerate(r['scores'])
                          if r['items'][i]['kind']==kind for j,s in enumerate(v[g]) if abs(DOSES[j])==magnitude]
                    delta.append(max([0.]+vals))
                curves.append(dict(layers=[x+1 for x in GROUPS[g]],magnitude=magnitude,
                                   oracle_mean_ce_reduction=float(np.mean(delta))))
        report['dose_layer_curve']=curves; result[kind]=report
    mass=[p for r in records for p in r['attention_mass']]
    result['attention_probe']=dict(measurements=len(mass),
        expected_direction=sum((p['changed']['mass']-p['zero']['mass'])*p['dose']>=-1e-7 for p in mass),
        mean_absolute_mass_change=float(np.mean([abs(p['changed']['mass']-p['zero']['mass']) for p in mass]))) if mass else None
    if mass and result['attention_probe']['expected_direction']!=len(mass):
        raise RuntimeError('Measured attention mass did not follow the injected dose; inspect implementation before training')
    matched=[(r['items'][i],r['items'][i+1]) for r in records for i in range(0,len(r['items']),2)]
    result['matching_balance']=dict(pairs=len(matched),
        covariates=['log_receiver_length','log_sender_length','relative_token_distance',
                    'receiver_rank','sender_rank','receiver_question_overlap','sender_question_overlap','same_document'],
        mean_absolute_difference=np.mean([np.abs(np.asarray(x['covariates'])-y['covariates'])
                                          for x,y in matched],0).tolist() if matched else None,
        limitation='Approximate covariate matching; a graph/non-graph difference is not randomized topology evidence.')
    lookup={r['sample_id']:r for r in records}
    order=[]
    for r in reversed_records:
        orig=lookup[r['sample_id']]; x,y=gains(orig),gains(r)
        strong=(np.abs(x)>a.effect_floor)|(np.abs(y)>a.effect_floor)
        order.append(dict(sample_id=r['sample_id'],original_base=orig['base'],reversed_base=r['base'],
                          informative_actions=int(strong.sum()),sign_changes=int(((x*y<0)&strong).sum()),
                          max_gain_original=float(max(0.,x.max())) if x.size else 0.,
                          max_gain_reversed=float(max(0.,y.max())) if y.size else 0.))
    result['order_diagnostic']=dict(n=len(order),details=order,
        limitation='Both token order and allowed information-flow direction change; this does not isolate either one.')
    result['passed']=bool(result['graph']['available_accuracy_delta_pp']>=4 and result['graph']['ce_reduction']>=.01)
    result['next']='Effect prediction pilot' if result['passed'] else 'Stop: sampled intervention family has insufficient accuracy headroom; no predictor training.'
    return result


@torch.no_grad()
def predictions(model,record,scale):
    return model(record['features']).cpu().numpy()*scale


def policy_metrics(model,records,scale,floor,constant):
    out=[]
    for r in records:
        pred=predictions(model,r,scale); true=gains(r)
        learned=chosen_gain(r,pred,'graph',floor)
        const=chosen_gain(r,np.broadcast_to(constant,true.shape),'graph',floor)
        oracle=max([0.]+[float(v.max()) for i,v in enumerate(true) if r['items'][i]['kind']=='graph'])
        out.append(dict(sample_id=r['sample_id'],learned_gain=learned,constant_gain=const,
                        oracle_gain=oracle,regret=oracle-learned))
    return dict(n=len(out),learned_gain=float(np.mean([r['learned_gain'] for r in out])),
                constant_gain=float(np.mean([r['constant_gain'] for r in out])),
                oracle_gain=float(np.mean([r['oracle_gain'] for r in out])),
                regret=float(np.mean([r['regret'] for r in out]))),out


def train_step(model,opt,records,scale,floor):
    opt.zero_grad(set_to_none=True)
    loss=[]
    for r in records:
        if not r['items']: continue
        pred=model(r['features'])
        target=torch.tensor(gains(r),device=pred.device,dtype=torch.float32)
        weights=1+(target.abs()>floor).float()
        loss.append((F.smooth_l1_loss(pred,target/scale,reduction='none')*weights).mean())
    if not loss:return 0.
    value=torch.stack(loss).mean()
    if not torch.isfinite(value): raise RuntimeError('Nonfinite predictor loss')
    value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
    return float(value.detach())


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(a,train,dev,h,device,progress):
    informative=[r for r in train if gains(r).size and gains(r).max()>a.effect_floor]
    values=np.concatenate([gains(r).reshape(-1) for r in train if r['items']]) if informative else np.array([0.])
    scale=max(.01,float(np.quantile(np.abs(values),.9)))
    # Each question, not each edge, gets equal weight in the constant baseline.
    constant=np.mean([gains(r).mean(0) for r in train if r['items']],0) if informative else np.zeros(len(ACTIONS))
    dim=next(r['features']['x'].shape[-1] for r in train)
    def fresh():
        torch.manual_seed(a.seed); return EffectPredictor(dim).to(device)
    tiny_path=a.output_root/'predictor/TINY.json'
    progress.stage(8,a.tiny_steps,unit='step',kind='fit')
    if tiny_path.exists():
        tiny=json.loads(tiny_path.read_text())
        if tiny['contract_sha256']!=h:raise RuntimeError('Tiny contract mismatch')
    elif len(informative)<8:
        tiny=dict(contract_sha256=h,passed=False,reason='Fewer than eight informative training questions')
        atomic_json(tiny_path,tiny)
    else:
        tiny_rows=sorted(informative,key=lambda r:stable_key(a.seed,'tiny',r['sample_id']))[:16]
        model=fresh(); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
        for step in range(a.tiny_steps):
            batch=random.Random(a.seed+step).sample(tiny_rows,min(8,len(tiny_rows)))
            train_step(model,opt,batch,scale,a.effect_floor); progress.update()
        metric,_=policy_metrics(model,tiny_rows,scale,a.effect_floor,constant)
        tiny=dict(contract_sha256=h,n=len(tiny_rows),sample_ids=[r['sample_id'] for r in tiny_rows],
                  metric=metric,passed=bool(metric['oracle_gain']>0 and metric['learned_gain']>=.8*metric['oracle_gain']))
        atomic_json(tiny_path,tiny)
    progress.complete(f'passed={tiny["passed"]}; {tiny_path}')
    if not tiny['passed']:return None,scale,constant,tiny
    model=fresh(); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
    root=a.output_root/'predictor'; last=root/'last.pt'; best_path=root/'best.pt'
    start=0; history=[]
    if last.exists():
        saved=read_tensor(last,h); model.load_state_dict(saved['state']); opt.load_state_dict(saved['optimizer'])
        for state in opt.state.values():
            for key,v in state.items():
                if isinstance(v,torch.Tensor): state[key]=v.to(device)
        start=saved['epoch']; history=saved['history']; best=read_tensor(best_path,h)
    else:
        metric,_=policy_metrics(model,dev,scale,a.effect_floor,constant)
        best=dict(contract_sha256=h,state=cpu_state(model),epoch=0,metric=metric,scale=scale,constant=constant.tolist())
        save_tensor(best_path,best)
    progress.stage(9,a.epochs*len(train),initial=start*len(train),unit='q',kind='fit')
    for epoch in range(start,a.epochs):
        epoch_started=time.monotonic()
        progress.names[9]=f'B: predictor epoch {epoch+1}/{a.epochs}'
        order=list(train); random.Random(a.seed+epoch).shuffle(order)
        losses=[]
        for begin in range(0,len(order),8):
            batch=order[begin:begin+8]
            losses.append(train_step(model,opt,batch,scale,a.effect_floor)); progress.update(len(batch))
        train_metric,_=policy_metrics(model,train,scale,a.effect_floor,constant)
        metric,_=policy_metrics(model,dev,scale,a.effect_floor,constant)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),train=train_metric,dev=metric))
        if metric['regret']<best['metric']['regret']:
            best=dict(contract_sha256=h,state=cpu_state(model),epoch=epoch+1,metric=metric,scale=scale,constant=constant.tolist())
            save_tensor(best_path,best)
        save_tensor(last,dict(contract_sha256=h,state=cpu_state(model),optimizer=opt.state_dict(),
                             epoch=epoch+1,history=history))
        atomic_json(root/'history.json',history)
        progress.weights[9]=(time.monotonic()-epoch_started)*a.epochs
    model.load_state_dict(best['state']); model.eval()
    progress.complete(f'best epoch={best["epoch"]}; {best_path}')
    return model,scale,constant,tiny


def joint(a,w,rows,records,pad,h,model,scale,constant,progress):
    modes=('zero','constant','random','graph1','graph2','non_graph1')
    ch=sha_file(a.output_root/'predictor/best.pt')
    def path(row,mode):return a.output_root/'joint'/mode/(stable_key(0,row['sample_id'])+'.json')
    cached=sum(path(row,m).exists() for row in rows for m in modes)
    progress.stage(11,len(rows)*len(modes),initial=cached,unit='forward',kind='validation')
    output=[]
    for row,r in zip(rows,records,strict=True):
        pred=predictions(model,r,scale); actual=gains(r)
        for mode in modes:
            out=path(row,mode)
            if out.exists():
                record=json.loads(out.read_text())
                if record['contract_sha256']!=h or record['checkpoint_sha256']!=ch:
                    raise RuntimeError('Joint output contract/checkpoint mismatch')
            else:
                if mode=='zero': selected=[]
                elif mode=='constant': selected=select_actions(np.broadcast_to(constant,actual.shape),r['items'],'graph',1,a.effect_floor)
                elif mode=='random':
                    choices=[None]+[(i,j) for i,x in enumerate(r['items']) if x['kind']=='graph' for j in range(len(ACTIONS))]
                    which=random.Random(stable_key(a.seed,'random',row['sample_id'])).choice(choices)
                    selected=[] if which is None else [which]
                else:
                    kind='non_graph' if mode=='non_graph1' else 'graph'
                    selected=select_actions(pred,r['items'],kind,2 if mode=='graph2' else 1,a.effect_floor)
                set_actions(w,[(i,*ACTIONS[j]) for i,j in selected])
                outcome=score(w,candidate_row(row,r['items']),pad)
                additive=sum(float(actual[i,j]) for i,j in selected)
                record=dict(contract_sha256=h,checkpoint_sha256=ch,sample_id=row['sample_id'],
                            dataset=row['dataset'],mode=mode,selected=[list(v) for v in selected],score=outcome,
                            predicted_gains=[float(pred[i,j]) for i,j in selected],
                            measured_single_gains=[float(actual[i,j]) for i,j in selected],
                            additive_single_gain=additive,joint_gain=r['base']['ce']-outcome['ce'],
                            interaction_residual=r['base']['ce']-outcome['ce']-additive)
                if mode=='zero' and abs(outcome['ce']-r['base']['ce'])>a.noise_tolerance:
                    raise RuntimeError('Joint zero baseline differs from intervention measurement')
                atomic_json(out,record); progress.update()
            output.append(record)
    progress.complete(f'{len(output)} predictions; {a.output_root/"joint"}')
    atomic_jsonl(a.output_root/'joint_predictions.jsonl',output)
    return output


def final_report(a,records,output,model,scale,constant,capability,tiny,h):
    bymode={m:[x for x in output if x['mode']==m] for m in ('zero','constant','random','graph1','graph2','non_graph1')}
    baseline=[x['score'] for x in bymode['zero']]; ds=[x['dataset'] for x in bymode['zero']]
    qa={m:compare_scores(baseline,[x['score'] for x in rows],ds) for m,rows in bymode.items()}
    local,details=policy_metrics(model,records,scale,a.effect_floor,constant)
    advantage=[r['learned_gain']-r['constant_gain'] for r in details]
    constant_ci=paired_interval(advantage)
    passed=bool(qa['graph1']['ce_reduction']>.01 and qa['graph1']['ce_reduction_ci95'][0]>0
                and float(np.mean(advantage))>.01 and constant_ci[0]>0 and qa['graph1']['accuracy_delta_pp']>=2)
    two=[r for r in bymode['graph2'] if len(r['selected'])==2]
    diagnoses=[]
    if tiny['passed'] and local['learned_gain']<=max(0.,local['constant_gain']):
        diagnoses.append('Train16 can fit, but heldout policy does not beat zero/global action: transfer or feature-target identifiability remains unresolved; capacity vs data size is not proven.')
    if qa['graph1']['ce_reduction']>qa['graph2']['ce_reduction']:
        diagnoses.append('Top2 is worse than Top1; inspect per-question interaction residuals before scaling simultaneous corrections.')
    if qa['non_graph1']['ce_reduction']>=qa['graph1']['ce_reduction']:
        diagnoses.append('Matched non-KG policy is at least as good in mean CE; current pilot does not establish KG-specific benefit.')
    if not passed:
        diagnoses.append('Stop before full training. This failure applies to the tested candidates/features/action family, not all graph-attention methods.')
    return dict(contract_sha256=h,status='complete',scope='Official train-resplit exploratory pilot; no TEST or original Val1000 evaluation.',
                capability=capability,tiny=tiny,local=local,local_questions=details,
                learned_vs_constant_gain_ci95=constant_ci,qa=qa,
                joint_interaction=dict(n=len(two),mean_residual=float(np.mean([r['interaction_residual'] for r in two])) if two else 0.,
                                       negative_residual=sum(r['interaction_residual']<-a.effect_floor for r in two)),
                passed=passed,diagnoses=diagnoses,
                next='Independent larger validation replication; do not claim full TEST gain.' if passed else 'Review diagnoses; no automatic scaling.')


def safety(a,w,probe,row,items,pad,h):
    model_row=candidate_row(row,items)
    set_actions(w,[]); w.controller.capture=True
    first=score(w,model_row,pad); f=w.controller.captured
    repeat=score(w,model_row,pad)
    changed=dict(model_row,input_ids=list(model_row['input_ids']),gold_index=(model_row['gold_index']+1)%4)
    end=changed['question_span'][1]
    changed['input_ids'][end:]=[pad]*(len(changed['input_ids'])-end)
    score(w,changed,pad); other=w.controller.captured
    for name in ('q','p'):torch.testing.assert_close(f[name],other[name],atol=0,rtol=0)
    w.controller.capture=False; w.controller.captured=None
    if abs(first['ce']-repeat['ce'])>a.noise_tolerance or first['prediction']!=repeat['prediction']:
        raise RuntimeError('Repeated baseline is unstable')
    if any(p.requires_grad for p in w.llm.parameters()):raise RuntimeError('Target weights must be frozen')
    with torch.no_grad(),training_attention_backend():
        expected=w.zero_bias_reference(model_row)
        actual=w([model_row],pad_id=pad).last_hidden
    torch.testing.assert_close(actual,expected,atol=.03,rtol=.02)
    atomic_json(a.output_root/'SAFETY.json',dict(contract_sha256=h,target_frozen=True,
                excluded_options_and_gold_from_features=True,repeat_ce_delta=abs(first['ce']-repeat['ce']),
                native_same_mask_parity=True,gpu=torch.cuda.get_device_name(0)))


def run(a,rows,plans,contract,h,progress):
    groups={s:[r for r in rows if r['diagnostic_split']==s] for s in ('capability','train','dev','heldout')}
    progress.stage(0,1)
    mp=a.output_root/'manifest.json'
    if not mp.exists():
        provenance={k:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
                    for k,args in [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current'])]}
        atomic_json(mp,dict(contract=contract,contract_sha256=h,git=provenance))
    atomic_jsonl(a.output_root/'cohort.jsonl',contract['cohort'])
    progress.complete(f'fresh official TRAIN only; original Val1000/TEST untouched; {mp}')
    final=a.output_root/'SUMMARY.json'
    if final.exists():
        r=json.loads(final.read_text())
        if r['contract_sha256']!=h:raise RuntimeError('Summary mismatch')
        return r
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available():
        raise RuntimeError('Run launcher on the server with working GPU1; no GPU was allocated')
    progress.stage(1,1)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error(); logging.disable_progress_bar()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,
                                          device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    c=InterventionController(llm.config.hidden_size,layers=12).to('cuda')
    w=PropositionAttentionModel(llm,c,attention_layers=tuple(i for g in GROUPS for i in g),gradient_checkpointing=False).eval()
    probe=AttentionProbe();install_probe(probe)
    progress.complete('pristine Llama; all weights frozen; same enabled 12-layer mask schedule')
    progress.stage(2,1)
    sample=next(r for r in rows if plans[r['sample_id']])
    safety(a,w,probe,sample,plans[sample['sample_id']],pad,h)
    progress.complete(f'{a.output_root/"SAFETY.json"}')
    cap=measure(a,w,probe,groups['capability'],plans,pad,h,progress,3)
    reverse=measure(a,w,probe,subset(groups['capability'],a.order_questions,a.seed),plans,pad,h,progress,4,'reversed')
    progress.stage(5,1)
    report=capability_report(a,cap,reverse,h)
    atomic_json(a.output_root/'CAPABILITY.json',report)
    atomic_jsonl(a.output_root/'capability_interventions.jsonl',[audit_record(r) for r in cap])
    progress.complete(f'passed={report["passed"]}; {a.output_root/"CAPABILITY.json"}')
    if not report['passed']:
        return dict(contract_sha256=h,status='stopped_capability_gate',capability=report,passed=False)
    if a.phase=='capability':
        return dict(contract_sha256=h,status='capability_only',capability=report)
    train=measure(a,w,probe,groups['train'],plans,pad,h,progress,6)
    dev=measure(a,w,probe,groups['dev'],plans,pad,h,progress,7)
    atomic_jsonl(a.output_root/'train_dev_interventions.jsonl',[audit_record(r) for r in train+dev])
    model,scale,constant,tiny=fit(a,train,dev,h,llm.device,progress)
    if model is None:
        return dict(contract_sha256=h,status='stopped_tiny_overfit',capability=report,tiny=tiny,passed=False)
    heldout=measure(a,w,probe,groups['heldout'],plans,pad,h,progress,10)
    atomic_jsonl(a.output_root/'heldout_interventions.jsonl',[audit_record(r) for r in heldout])
    output=joint(a,w,groups['heldout'],heldout,pad,h,model,scale,constant,progress)
    progress.stage(12,1)
    result=final_report(a,heldout,output,model,scale,constant,report,tiny,h)
    progress.complete(f'passed={result["passed"]}; {final}')
    return result


def main(argv=None):
    a=arguments(argv);torch.set_num_threads(8)
    rows,plans,contract,h=preflight(a)
    counts=Counter((r['diagnostic_split'],r['dataset']) for r in rows)
    active=[r for r in rows if a.phase=='all' or r['diagnostic_split']=='capability']
    forwards=sum(2+len(plans[r['sample_id']])*len(ACTIONS) for r in active)
    if a.preflight_only:
        print(json.dumps(dict(status='preflight_ok',contract_sha256=h,
            counts={f'{s}/{d}':n for (s,d),n in counts.items()},
            graphless_or_unmatched=sum(not plans[r['sample_id']] for r in rows),
            measurement_forwards_before_order_safety_joint=forwards,
            cached=sum(cache_path(a,r).exists() for r in active),max_tokens=max(len(r['input_ids']) for r in rows),
            output=str(a.output_root),no_model_loaded=True,no_job_started=True),indent=2));return
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=Progress(list(STAGES[:6] if a.phase=='capability' else STAGES))
        # Explicit small CPU-report cost estimates; model/measurement/training
        # stages remain unknown until their own fresh operation is calibrated.
        progress.weights[5]=1.
        if len(progress.names)>12:progress.weights[12]=1.
        try:
            progress.log(f'[plan] A {a.capability_questions} fresh questions: dose/layer sweep + matched non-KG + order{a.order_questions}; gated B Train{a.train_questions}/Dev{a.dev_questions}; C Heldout{a.heldout_questions}: top1/top2; phase={a.phase}; <= {forwards:,} main measurement forwards')
            progress.log('[ETA] calibrating per operation on fresh questions; all comparisons batch1, 12 controlled layers. Previous 4-layer timings are not a compatible ETA. Completed question/epoch caches resume.')
            result=run(a,rows,plans,contract,h,progress)
            path=a.output_root/('CAPABILITY_ONLY.json' if result['status']=='capability_only' else 'SUMMARY.json')
            atomic_json(path,result)
            for i in range(progress.index+1,len(progress.names)):
                progress.stage(i,1,initial=1);progress.complete(f'not executed: {result["status"]}')
            progress.log(f'[result] status={result["status"]}; passed={result.get("passed","not evaluated")}; {path}')
        except Exception as e:
            progress.log(f'[FAILED] stage={progress.index+1}/{len(progress.names)} item={progress.done}/{progress.total}; {type(e).__name__}: {e}; durable={a.output_root}; identical command resumes complete questions/epochs')
            raise
        finally:progress.close()


if __name__=='__main__':main()

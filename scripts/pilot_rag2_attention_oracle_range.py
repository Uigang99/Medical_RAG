#!/usr/bin/env python3
"""Fixed-seed amplitude versus optimization-range/time oracle diagnostic.

Reuses ALL 32 official-TRAIN errors and their saved best pair/group tables.
No test, generalizing controller, new retrieval, text reordering or Llama training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts import pilot_rag2_attention_oracle as source
from medrag.training.attention_oracle_range import (
    SCALES, BOUNDS, check_seed, new_state, observe, milestone, summarize,
)
from medrag.training.attention_oracle import score, optimize_step
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, sha_file
from medrag.training.graph_input_progress import Progress

VERSION = 'attention_oracle_range_v1'
STAGES = ['validate frozen TRAIN32', 'load frozen Llama', 'parity / gradient safety',
          'scale-only / replay', 'joint bound2', 'joint bound4', 'joint bound8', 'paired range/time summary']
CODE = source.CODE + ['scripts/pilot_rag2_attention_oracle_range.py',
    'scripts/run_rag2_attention_oracle_range_pilot.sh', 'medrag/training/attention_oracle_range.py']


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=ROOT/'results/rag2_attention_oracle_v1')
    p.add_argument('--discovery-root', type=Path, default=ROOT/'results/rag2_attention_identifiability_v1')
    p.add_argument('--prepared-root', type=Path, default=ROOT/'results/rag2_official_train_proposition_attention_v1')
    p.add_argument('--output-root', type=Path, default=ROOT/'results/rag2_attention_oracle_range_v1')
    p.add_argument('--steps', type=int, default=80, help='Additional updates per bound; compare halfway and final')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if a.steps < 2 or a.steps % 2:
        p.error('--steps must be a positive even number >=2')
    return a


def preflight(a):
    for root in (a.source_root, a.discovery_root, a.prepared_root):
        if a.output_root.resolve() == root.resolve() or root.resolve() in a.output_root.resolve().parents:
            raise RuntimeError('Use a separate output root; source artifacts are immutable')
    manifest = json.loads((a.source_root/'manifest.json').read_text())
    sh = manifest['contract_sha256']
    if digest(manifest['contract']) != sh:
        raise RuntimeError('Corrupt source oracle manifest')
    report = source.read_json(a.source_root/'SUMMARY.json', sh)
    if report['status'] != 'complete' or report['n'] != 32 or report['missing'] != 0:
        raise RuntimeError('Require complete frozen oracle32, with no missing questions')
    # Reuse the original exact-input/model/code contract validator. No retokenization.
    settings = manifest['contract']['settings']
    old_args = SimpleNamespace(source_root=a.discovery_root, prepared_root=a.prepared_root,
                              output_root=a.source_root, preflight_only=True, **settings)
    rows, _, old_contract, verified = source.preflight(old_args)
    if verified != sh or old_contract != manifest['contract']:
        raise RuntimeError('Source code, environment, model, or cohort changed; do not mix scores')
    seeds, baselines, files = {}, {}, {}
    for row in rows:
        directory = source.qdir(old_args, row)
        seed = source.read_json(directory/'joint.json', sh)
        check_seed(row, seed)
        if seed['groups_one_based'] != [[i+1 for i in g] for g in source.GROUPS]:
            raise RuntimeError('Layer grouping changed')
        base = source.read_json(directory/'baseline.json', sh)
        if base['sample_id'] != row['sample_id'] or base['row_sha256'] != digest(row):
            raise RuntimeError('Baseline question or input hash changed')
        for result in (base['score'], seed['best']):
            if result['gold'] != row['gold_index'] or not torch.isfinite(torch.tensor(result['logits'])).all():
                raise RuntimeError('Invalid source scoring record')
        seeds[row['sample_id']], baselines[row['sample_id']] = seed, base['score']
        for name in ('joint.json', 'baseline.json'):
            files[str(directory/name)] = sha_file(directory/name)
    contract = dict(version=VERSION, source_contract_sha256=sh, source_files=files,
        model=old_contract['model'], source_environment=old_contract['environment'],
        cohort=old_contract['cohort'], source_prepared_sha256=old_contract['prepared_sha256'],
        settings=dict(scales=list(SCALES), bounds=list(BOUNDS), steps=a.steps,
                      learning_rate=settings['learning_rate'], query_chunk=settings['query_chunk'], seed=settings['seed']),
        code={p:sha_file(ROOT/p) for p in CODE},
        scientific=dict(
            purpose='Separate fixed-pattern amplitude, extra search time, and expanded joint-control range.',
            input='Identical 32 official TRAIN errors, source tokens/order/top8/KG pairs and frozen pristine Llama.',
            initialization='Every bound starts from the SAME saved best table, with fresh Adam moments. Scale-only scores NEVER initialize optimization.',
            operation='All causal pair x 3 group scalars optimized jointly per question; target model frozen. No reverse causal attention.',
            loss='Four-choice gold-label CE. Per-question oracle ONLY; gold is not available for real inference.',
            checkpoints=[0,a.steps//2,a.steps], selection='Best observed CE including the common seed; raw endpoints also saved.',
            primary='Error recovery counts at equal steps for bounds 2/4/8; scale-only raw recovery counts.',
            secondary='CE, gold probability, margin, paired transitions, saturation and gradient/update diagnostics.',
            screen='At least 4/32 recovered and mean CE reduction >=.05 versus no bias; wider range must separately improve over bound2 at equal steps.',
            stop='All three bounds run regardless of interim performance; no automatic extension beyond range8 or configured updates.',
            scope='Gold-conditioned error-selected MCQ capacity diagnosis; NOT held-out accuracy, generalization, or a proven global optimum.'))
    h = digest(contract)
    mp = a.output_root/'manifest.json'
    if mp.exists() and source.read_json(mp,h)['contract'] != contract:
        raise RuntimeError('Output contract changed; choose a new --output-root')
    return rows, seeds, baselines, contract, h


def replay_check(actual, expected, label):
    delta = max(abs(x-y) for x,y in zip(actual['logits'],expected['logits'],strict=True))
    if delta > 1e-4 or abs(actual['ce']-expected['ce']) > 1e-4:
        raise RuntimeError(f'{label} no longer reproduces saved scores (logit delta={delta}); stop before comparing bounds')


def scale_only(a, w, rows, seeds, baselines, pad, h, progress):
    cached = sum((source.qdir(a,r)/'scaling.json').exists() for r in rows)
    progress.stage(3,len(rows)*5,initial=cached*5,unit='forward',kind='validation')
    result = {}
    for row in rows:
        path = source.qdir(a,row)/'scaling.json'; i = row['sample_id']
        if path.exists():
            result[i] = source.read_json(path,h); continue
        source.attach(w,row)
        zero = score(w,row,pad); replay_check(zero,baselines[i],'zero')
        values = check_seed(row,seeds[i]); outputs = {}
        for scale in SCALES:
            w.controller.assign(values*scale)
            outputs[f'{scale:g}'] = score(w,row,pad)
        replay_check(outputs['1'],seeds[i]['best'],'saved oracle')
        record = dict(contract_sha256=h,sample_id=i,zero=zero,scales=outputs,
                      seed_sha256=digest(seeds[i]),raw_scores_without_noop_selection=True)
        atomic_json(path,record); result[i]=record; progress.update(5)
    progress.complete(f'All scale-only raw scores saved, including harms; {a.output_root}/questions')
    return result


def bound_paths(a,row,bound):
    directory = source.qdir(a,row)/f'bound{bound:g}'
    return directory/'last.pt', directory/'result.json'


def optimize_bound(a,w,row,seed,pad,h,bound,lr,progress):
    checkpoint, final = bound_paths(a,row,bound)
    if final.exists():
        return source.read_json(final,h)
    source.attach(w,row)
    optimizer = torch.optim.Adam([w.controller.values],lr=lr)
    if checkpoint.exists():
        state=source.read_tensor(checkpoint,h)
        if state['sample_id'] != row['sample_id'] or state['bound'] != bound or state['initial_values'] != seed['values']:
            raise RuntimeError('Checkpoint arm/question/initialization mismatch')
        w.controller.assign(state['values']); optimizer.load_state_dict(state['optimizer'])
    else:
        values=check_seed(row,seed); w.controller.assign(values)
        state=new_state(h,row,seed,bound)
        milestone(state,seed['best'],values,0)
    if not state['complete']:
        for step in range(state['step'],a.steps):
            measured,before,diagnostics=optimize_step(w,row,pad,optimizer,bound)
            observe(state,measured,before,step)
            state['history'].append(dict(step_before_update=step,score_before_update=measured,**diagnostics))
            state['step']=step+1
            if state['step'] in (a.steps//2,a.steps):
                endpoint=score(w,row,pad)
                milestone(state,endpoint,w.controller.values.detach().cpu().clone(),state['step'])
            if state['step']%5==0 or state['step'] in (a.steps//2,a.steps):
                state.update(values=w.controller.values.detach().cpu().clone(),optimizer=optimizer.state_dict())
                source.save_tensor(checkpoint,state)
            progress.update()
        # Both exact endpoints have already been measured; no off-by-one extra update.
        state.update(complete=True,values=w.controller.values.detach().cpu().clone(),optimizer=optimizer.state_dict())
        source.save_tensor(checkpoint,state)
    w.controller.assign(torch.tensor(state['best_values']))
    replay_check(score(w,row,pad),state['best'],'selected range oracle')
    record={k:v for k,v in state.items() if k not in ('optimizer','values')}
    record.update(ordered_pairs=row['pairs'],groups_one_based=[[i+1 for i in g] for g in source.GROUPS],
                  endpoint_values=state['values'].tolist(),learning_rate=lr)
    atomic_json(final,record)
    return record


def cached_steps(a,rows,bound,h):
    total=0
    for row in rows:
        ck,final=bound_paths(a,row,bound)
        if final.exists():
            record=source.read_json(final,h)
            if not record['complete'] or record['step'] != a.steps:
                raise RuntimeError('Incomplete cached result')
            total+=a.steps
        elif ck.exists():
            total+=source.read_tensor(ck,h)['step']
    return total


def run(a,rows,seeds,baselines,contract,h,progress):
    mp=a.output_root/'manifest.json'
    if not mp.exists():
        git={k:subprocess.check_output(['git',*v],cwd=ROOT,text=True).strip() for k,v in
             [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current'])]}
        atomic_json(mp,dict(contract=contract,contract_sha256=h,git=git))
    atomic_jsonl(a.output_root/'cohort.jsonl',contract['cohort'])
    progress.complete(f'Frozen TRAIN32 / source best tables checked; {mp}')
    final=a.output_root/'SUMMARY.json'
    if final.exists():
        return source.read_json(final,h)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available():
        raise RuntimeError('Run on server GPU1; no model allocated')
    settings=contract['settings']; torch.manual_seed(settings['seed'])
    progress.stage(1,1)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error(); logging.disable_progress_bar()
    tok=AutoTokenizer.from_pretrained(contract['model'],local_files_only=True)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm=AutoModelForCausalLM.from_pretrained(contract['model'],local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    w=source.PropositionAttentionModel(llm,source.OracleBiasTable(llm.config.hidden_size,rows[0]['pairs']).to('cuda'),
        attention_layers=tuple(i for g in source.GROUPS for i in g),gradient_checkpointing=True,
        query_chunk=settings['query_chunk']).eval()
    source.install_oracle_attention()
    progress.complete('Pristine frozen Llama; only current question bias table is trainable')
    progress.stage(2,1)
    check=source.safety(w,max(rows,key=lambda r:len(r['input_ids'])),pad)
    check.update(contract_sha256=h,gpu=torch.cuda.get_device_name(0),peak_vram_bytes=torch.cuda.max_memory_allocated())
    atomic_json(a.output_root/'SAFETY.json',check)
    progress.weights[3]=check['calibration_forward_seconds']*len(rows)*5
    for j in (4,5,6):
        progress.weights[j]=len(rows)*(a.steps*check['calibration_backward_seconds']+3*check['calibration_forward_seconds'])
    progress.weights[7]=2.
    progress.complete('Grad/no-grad parity and backward checked; forward/update ETA calibrated separately')
    scaling=scale_only(a,w,rows,seeds,baselines,pad,h,progress)
    optimized={r['sample_id']:{} for r in rows}
    for j,bound in enumerate(BOUNDS,4):
        cached=cached_steps(a,rows,bound,h)
        progress.stage(j,len(rows)*a.steps,initial=cached,unit='update',kind='train')
        for row in rows:
            progress.names[j]=f'bound{bound:g} {row["dataset"]} joint'
            optimized[row['sample_id']][bound]=optimize_bound(a,w,row,seeds[row['sample_id']],pad,h,bound,settings['learning_rate'],progress)
        progress.complete(f'Bound +/-{bound:g}: same source seed, fresh Adam per question; {a.output_root}/questions')
    progress.stage(7,1)
    report=summarize(rows,seeds,scaling,optimized,a.steps,h)
    atomic_jsonl(a.output_root/'per_question.jsonl',report['per_question'])
    atomic_json(final,report)
    progress.complete(f'Paired scale/range/time contrasts and raw endpoints saved; {final}')
    return report


def main(argv=None):
    a=arguments(argv); torch.set_num_threads(8)
    progress=None
    try:
        if not a.preflight_only:
            progress=Progress(list(STAGES)); progress.stage(0,1)
        rows,seeds,baselines,contract,h=preflight(a)
        counts=dict(questions=len(rows),datasets=dict(Counter(r['dataset'] for r in rows)),
            pairs=sum(len(r['pairs']) for r in rows),scale_forwards=len(rows)*len(SCALES),
            zero_replays=len(rows),optimizer_updates=len(rows)*len(BOUNDS)*a.steps,
            cached_updates={f'{b:g}':cached_steps(a,rows,b,h) for b in BOUNDS},
            max_tokens=max(len(r['input_ids']) for r in rows))
        if a.preflight_only:
            print(json.dumps(dict(status='preflight_ok',contract_sha256=h,**counts,
                model_loaded=False,gpu_job_started=False,output=str(a.output_root)),indent=2)); return
        progress.log(f'[plan] frozen TRAIN32 -> frozen Llama -> safety -> scales .5/1/2/4 -> bounds 2/4/8, each {a.steps} updates -> paired report; {counts}')
        progress.log('[control] Same saved best seed and fresh Adam per bound; fixed text/order/layers; no causal reversal; no repeated single-pair sweep; no performance gate skips arms.')
        progress.log('[ETA] Prior 2,560 joint updates took about 22 min; default 7,680 updates roughly 1-1.5h on GPU1. Fresh forward/backward calibration refines this; checkpoint every 5 updates.')
        a.output_root.mkdir(parents=True,exist_ok=True)
        with (a.output_root/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=run(a,rows,seeds,baselines,contract,h,progress)
        if progress.index<7:
            for j in range(progress.index+1,8):
                progress.stage(j,1,initial=1);progress.complete('cached complete diagnostic')
        progress.log(f'[result] {result["status"]}; capacity screens={result["capacity_screen_by_bound"]}; {a.output_root}/SUMMARY.json; NOT test/generalization accuracy')
    except Exception as error:
        if progress:
            progress.log(f'[FAILED] stage={progress.index+1}/8 item={progress.done}/{progress.total} remaining={progress.total-progress.done}; {type(error).__name__}: {error}; durable={a.output_root}; identical command resumes cached questions/last optimizer checkpoint. Source results preserved.')
        raise
    finally:
        if progress: progress.close()


if __name__=='__main__': main()

#!/usr/bin/env python3
"""Oracle capacity diagnostic on ALL 32 prior capability errors (official TRAIN).

1. Re-score all causal KG pairs x 3 layer groups x 6 doses, with no candidate cap.
2. Measure raw simultaneous top2/4/8 interventions (including harmful results).
3. Independently optimize every pair/group scalar for EACH SAME question using
   its gold answer, 2 starts x 40 Adam steps, projected to [-2,2]. No gate skips 2/3.

This is label-conditioned, error-selected, MCQ-only diagnosis, not test accuracy,
learned routing, deployable inference or a global mathematical upper bound.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from medrag.generation.proposition_kg_attention import PropositionAttentionModel, validate_layout
from medrag.training.attention_intervention_diagnostic import ACTIONS, GROUPS, DOSES, compare_scores
from medrag.training.attention_oracle import (
    OracleBiasTable, install_oracle_attention, score, loss_and_logits,
    outcome, single_table, single_summary, additive_candidates, optimize_step,
)
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, identity, jsonl, sha_file
from medrag.training.graph_input_modules import training_attention_backend
from medrag.training.graph_input_progress import Progress
from medrag.training.proposition_pair_effects import stable_key
from scripts.pilot_rag2_attention_identifiability import save_tensor, read_tensor

VERSION = 'attention_oracle_v1'
STAGES = ['contract / errors32', 'load frozen Llama', 'numerical / gradient safety',
          'all-pair single sweep', 'raw top2/4/8 joint', 'question-local optimization', 'paired diagnosis']
CODE = ['scripts/pilot_rag2_attention_oracle.py', 'scripts/run_rag2_attention_oracle_pilot.sh',
        'medrag/training/attention_oracle.py', 'scripts/pilot_rag2_attention_identifiability.py',
        'medrag/training/attention_intervention_diagnostic.py',
        'medrag/generation/proposition_kg_attention.py', 'medrag/training/proposition_pair_effects.py',
        'medrag/training/graph_input_data.py', 'medrag/training/graph_input_modules.py',
        'medrag/training/graph_input_progress.py']


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=ROOT/'results/rag2_attention_identifiability_v1')
    p.add_argument('--prepared-root', type=Path, default=ROOT/'results/rag2_official_train_proposition_attention_v1')
    p.add_argument('--output-root', type=Path, default=ROOT/'results/rag2_attention_oracle_v1')
    p.add_argument('--joint-steps', type=int, default=40)
    p.add_argument('--learning-rate', type=float, default=.1)
    p.add_argument('--seed', type=int, default=314159)
    p.add_argument('--query-chunk', type=int, default=128)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if a.joint_steps < 1 or a.learning_rate <= 0 or a.query_chunk < 1:
        p.error('Positive steps, learning rate and query chunk required')
    return a


def preflight(a):
    source = json.loads((a.source_root/'manifest.json').read_text())
    if digest(source['contract']) != source['contract_sha256']:
        raise RuntimeError('Source diagnostic manifest is corrupt')
    original = list(jsonl(a.source_root/'capability_interventions.jsonl'))
    if len(original) != 128 or len({r['sample_id'] for r in original}) != 128:
        raise RuntimeError('Require complete unique original capability128, not a replacement cohort')
    if any(r['contract_sha256'] != source['contract_sha256'] for r in original):
        raise RuntimeError('Source prediction contract mismatch')
    errors = {r['sample_id']: r for r in original if not r['base']['correct']}
    if len(errors) != 32:
        raise RuntimeError('Expected all 32 original errors; refuse silent cohort substitution')
    prepared = json.loads((a.prepared_root/'manifest.json').read_text())
    complete = json.loads((a.prepared_root/'prepared/COMPLETE.json').read_text())
    rows_path = a.prepared_root/'prepared/rows.jsonl'
    if digest(prepared['contract']) != prepared['contract_sha256'] or complete['contract_sha256'] != prepared['contract_sha256']:
        raise RuntimeError('Prepared metadata mismatch')
    if prepared != source['contract']['source_manifest']:
        raise RuntimeError('Prepared data differ from the previous diagnostic')
    if sha_file(rows_path) != complete['rows_sha256'] or complete['rows_sha256'] != source['contract']['prepared_sha256']:
        raise RuntimeError('Prepared content changed')
    for meta in prepared['contract']['inputs']['model_files'].values():
        actual = identity(meta['path'], full_hash='sha256' in meta)
        if any(actual.get(k) != v for k, v in meta.items()):
            raise RuntimeError(f'Model/tokenizer changed: {meta["path"]}')
    counts, rows = Counter(), []
    for row in jsonl(rows_path):
        counts[row['split']] += 1
        if row['sample_id'] not in errors:
            continue
        row = dict(row, diagnostic_split='capability')
        if row['split'] != 'train' or digest(row) != errors[row['sample_id']]['row_sha256']:
            raise RuntimeError('Question is not the identical official-TRAIN input')
        validate_layout(row)
        if len(set(row['choice_token_ids'])) != 4 or not 0 <= row['gold_index'] < 4:
            raise RuntimeError('Require original four-label answer scoring')
        if not row['pairs']:
            raise RuntimeError('Graphless error must be explicitly diagnosed, not silently excluded')
        rows.append(row)
    if counts != {'train': 9000, 'validation': 1000} or len(rows) != 32:
        raise RuntimeError('Official data/cohort incomplete')
    rows.sort(key=lambda r: stable_key(a.seed, r['sample_id']))
    contract = dict(version=VERSION, source_hash=source['contract_sha256'],
        source_predictions_sha256=sha_file(a.source_root/'capability_interventions.jsonl'),
        prepared_sha256=complete['rows_sha256'], model=source['contract']['model'],
        settings={k: v for k, v in vars(a).items() if k not in
                  ('source_root', 'prepared_root', 'output_root', 'preflight_only')},
        cohort=[dict(sample_id=r['sample_id'], dataset=r['dataset'], row_sha256=digest(r),
                     pairs=len(r['pairs']), tokens=len(r['input_ids']),
                     prior_sampled_pairs=[i['pair'] for i in errors[r['sample_id']]['items'] if i['kind']=='graph']) for r in rows],
        code={p: sha_file(ROOT/p) for p in CODE},
        environment=dict(torch=torch.__version__, transformers=__import__('transformers').__version__),
        scientific=dict(
            hypothesis='Candidate coverage or joint control can recover errors that sampled single-pair actions could not.',
            scope='All 32 original TRAIN errors; gold used per question. No Val1000, Test6545 or deployable predictor.',
            input='Identical cached tokens, Top8, order, proposition spans, KG causal pairs, single-token MCQ labels.',
            layers_one_based=[[i+1 for i in g] for g in GROUPS], doses=list(DOSES), bound=2.,
            granularity='One FP32 scalar per causal proposition pair and group; same scalar across four layers and all heads.',
            loss='Four-choice cross entropy; target Llama and all input data frozen; no LoRA or new model.',
            precision='BF16 Llama + FP32 selected lm_head and bias table; MATH for all controlled layers, both grad and no-grad.',
            change_from_prior='Forward backend and registered candidate mask set are harmonized for optimization. Recompute zero and old sampled actions; never pool old and new kernel scores.',
            search='Exhaustive finite SINGLE actions; raw positive top2/4/8 compositions; ALL pairs optimized jointly by projected Adam, 2 starts.',
            starts='Zero; best measured no-op/single/composition. If that is zero, deterministic uniform [-.05,.05].',
            gates='None between single/composition/optimization; only invalid numerics/contracts abort.',
            selection='Best observed CE including no-op; separately report any-correct and maximum margin encountered.',
            primary='Number of initially-wrong-under-current-backend questions corrected by best-CE joint result.',
            screen='At least 4 newly corrected questions AND mean CE reduction >=.05 on the error-selected cohort.',
            limits='Not representative accuracy, not held-out/generalization, not a KG-vs-nonKG ablation, not a global joint optimum. Failure cannot establish impossibility.',
            stop='Fixed 32 questions, 18 actions per pair, top2/4/8, 2 starts x configured steps; no automatic scaling.'))
    h = digest(contract)
    mp = a.output_root/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()).get('contract_sha256') != h:
        raise RuntimeError('Output contract differs; preserve it and choose a new --output-root')
    if a.output_root.resolve() in (a.source_root.resolve(), a.prepared_root.resolve()):
        raise RuntimeError('Output must not overwrite a source run')
    parent = a.output_root
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 2*1024**3:
        raise RuntimeError('Need at least 2 GiB free for durable oracle diagnostics')
    return rows, errors, contract, h


def qdir(a, row):
    return a.output_root/'questions'/stable_key(0, row['sample_id'])


def read_json(path, h):
    record = json.loads(path.read_text())
    if record['contract_sha256'] != h:
        raise RuntimeError(f'Cache contract mismatch: {path}')
    return record


def attach(w, row):
    w.controller = OracleBiasTable(w.llm.config.hidden_size, row['pairs']).to(w.llm.device)
    w.eval()


def safety(w, row, pad):
    attach(w, row)
    started = time.monotonic()
    first = score(w, row, pad)
    forward_seconds = time.monotonic()-started
    repeat = score(w, row, pad)
    with torch.no_grad(), training_attention_backend():
        native = w.zero_bias_reference(row)
        manual = w([row], pad_id=pad).last_hidden
    torch.testing.assert_close(manual, native, atol=.03, rtol=.02)
    w.train()
    with training_attention_backend():
        loss, logits = loss_and_logits(w, row, pad)
    grad_forward = outcome(loss, logits, row['gold_index'])
    del loss, logits
    delta = max(abs(x-y) for x, y in zip(first['logits'], grad_forward['logits'], strict=True))
    if abs(first['ce']-repeat['ce']) > 1e-4 or delta > 1e-4:
        raise RuntimeError(f'Gradient/no-gradient forward mismatch: {delta}; do not interpret oracle effects')
    opt = torch.optim.Adam([w.controller.values], lr=.1)
    started = time.monotonic()
    _, _, diagnostic = optimize_step(w, row, pad, opt, 2.)
    backward_seconds = time.monotonic()-started
    # Report finite differences without treating BF16 tiny differences as exact derivatives.
    gradient = w.controller.values.grad.detach().flatten()
    index = int(gradient.abs().argmax())
    probes = []
    for dose in (-.5, .5):
        values = torch.zeros_like(w.controller.values); values.flatten()[index] = dose
        w.controller.assign(values); probes.append(score(w, row, pad)['ce'])
    w.controller.assign(torch.zeros_like(w.controller.values)); w.eval()
    return dict(target_frozen=all(not p.requires_grad for p in w.llm.parameters()),
                native_zero_parity=True, repeat_ce_delta=abs(first['ce']-repeat['ce']),
                grad_forward_max_logit_delta=delta, optimizer_probe=diagnostic,
                finite_difference=dict(coordinate=index, step=.5, derivative=probes[1]-probes[0],
                                       analytic=float(gradient[index]), note='Coarse BF16 check, not a zero-noise derivative proof'),
                calibration_forward_seconds=forward_seconds, calibration_backward_seconds=backward_seconds)


def measure_singles(a, w, rows, old, pad, h, progress):
    total = sum(1+len(r['pairs'])*len(ACTIONS) for r in rows)
    cached = sum(int((qdir(a,r)/'baseline.json').exists()) + len(ACTIONS)*sum(
        (qdir(a,r)/f'singles/{i:05d}.json').exists() for i in range(len(r['pairs']))) for r in rows)
    progress.stage(3, total, initial=cached, unit='forward', kind='validation')
    results = {}
    for row in rows:
        attach(w, row); directory = qdir(a, row)
        path = directory/'baseline.json'
        if path.exists():
            baseline = read_json(path, h)
        else:
            baseline = dict(contract_sha256=h, sample_id=row['sample_id'], row_sha256=digest(row),
                            score=score(w,row,pad), prior_score=old[row['sample_id']]['base'])
            atomic_json(path, baseline); progress.update()
        records = []
        for i, pair in enumerate(row['pairs']):
            path = directory/f'singles/{i:05d}.json'
            if path.exists():
                record = read_json(path, h)
            else:
                scores = []
                for j in range(len(ACTIONS)):
                    w.controller.assign(single_table(len(row['pairs']), i, j))
                    scores.append(score(w, row, pad)); progress.update()
                record = dict(contract_sha256=h, sample_id=row['sample_id'], pair_index=i, pair=pair, scores=scores)
                atomic_json(path, record)
            if record['pair_index'] != i or record['pair'] != pair:
                raise RuntimeError('Single-action cached pair mismatch')
            records.append(record)
        w.controller.assign(torch.zeros_like(w.controller.values))
        zero = score(w, row, pad)
        if abs(zero['ce']-baseline['score']['ce']) > 1e-4:
            raise RuntimeError('Question zero baseline changed during sweep')
        summary = single_summary(baseline['score'], records,
                                 [i['pair'] for i in old[row['sample_id']]['items'] if i['kind']=='graph'])
        summary.update(contract_sha256=h, sample_id=row['sample_id'])
        atomic_json(directory/'single_summary.json', summary)
        results[row['sample_id']] = dict(base=baseline['score'], old=baseline['prior_score'], records=records, summary=summary)
    progress.complete(f'All {sum(len(r["pairs"]) for r in rows):,} KG pairs; durable per-pair scores; {a.output_root}/questions')
    return results


def measure_combinations(a, w, rows, singles, pad, h, progress):
    cached = sum((qdir(a,r)/f'combined_{k}.json').exists() for r in rows for k in (2,4,8))
    progress.stage(4, len(rows)*3, initial=cached, unit='forward', kind='validation')
    results = {}
    for row in rows:
        attach(w, row); single = singles[row['sample_id']]; output = []
        for plan in additive_candidates(single['base'], single['records'], len(row['pairs'])):
            path = qdir(a,row)/f'combined_{plan["k"]}.json'
            if path.exists():
                record = read_json(path, h)
            else:
                w.controller.assign(plan['values']); result = score(w,row,pad)
                gain = single['base']['ce']-result['ce']
                record = dict(contract_sha256=h, sample_id=row['sample_id'], score=result,
                    **{k:v for k,v in plan.items() if k!='values'}, values=plan['values'].tolist(),
                    actual_gain=gain, interaction_residual=gain-plan['additive_gain'])
                atomic_json(path,record); progress.update()
            output.append(record)
        results[row['sample_id']] = output
    progress.complete('Raw simultaneous results retained even when they harm CE; no oracle filtering of this control')
    return results


def initial_search(single, combined, n):
    best = single['summary']['exhaustive']
    values = torch.zeros(n, len(GROUPS)) if best['pair'] is None else single_table(n,best['pair'],best['action'])
    score0 = best['score']
    for record in combined:
        if record['score']['ce'] < score0['ce']:
            score0, values = record['score'], torch.tensor(record['values'])
    available = best['available_correct'] or any(r['score']['correct'] for r in combined)
    return score0, values, available


def optimize_question(a, w, row, single, combined, pad, h, progress):
    directory = qdir(a,row); final = directory/'joint.json'
    if final.exists():
        return read_json(final,h)
    attach(w,row)
    incumbent, incumbent_values, available = initial_search(single,combined,len(row['pairs']))
    starts = [torch.zeros_like(incumbent_values), incumbent_values.clone()]
    if not starts[1].any():
        gen = torch.Generator().manual_seed(int(stable_key(a.seed,row['sample_id'])[:12],16))
        starts[1] = (torch.rand(starts[1].shape,generator=gen)-.5)*.1
    restarts = []
    for restart, initial in enumerate(starts):
        progress.names[5] = f'joint {row["dataset"]} start {restart+1}/2'
        path = directory/f'optimization/start{restart}.pt'
        optimizer = torch.optim.Adam([w.controller.values],lr=a.learning_rate)
        if path.exists():
            state = read_tensor(path,h)
            w.controller.assign(state['values']); optimizer.load_state_dict(state['optimizer'])
        else:
            w.controller.assign(initial)
            state = dict(contract_sha256=h, sample_id=row['sample_id'], restart=restart, step=0,
                         best=incumbent, best_values=incumbent_values.clone(), available_correct=bool(available),
                         history=[], complete=False)
        if not state['complete']:
            for step in range(state['step'],a.joint_steps):
                measured, values, diagnostics = optimize_step(w,row,pad,optimizer,2.)
                if measured['ce'] < state['best']['ce']:
                    state['best'], state['best_values'] = measured,values
                state['available_correct'] |= measured['correct']
                state['history'].append(dict(step=step,score=measured,**diagnostics))
                state['step'] = step+1; progress.update()
                if (step+1)%5==0 or step+1==a.joint_steps:
                    state.update(values=w.controller.values.detach().cpu().clone(),optimizer=optimizer.state_dict())
                    save_tensor(path,state)
            w.eval(); last = score(w,row,pad)
            if last['ce'] < state['best']['ce']:
                state['best'],state['best_values'] = last,w.controller.values.detach().cpu().clone()
            state['available_correct'] |= last['correct']
            state.update(complete=True, final_score=last, values=w.controller.values.detach().cpu().clone(),optimizer=optimizer.state_dict())
            save_tensor(path,state)
        if state['best']['ce'] < incumbent['ce']:
            incumbent,incumbent_values = state['best'],state['best_values']
        available |= state['available_correct']
        restarts.append(dict(restart=restart, best=state['best'], final_score=state['final_score'],
                             history=state['history']))
    w.eval(); w.controller.assign(incumbent_values)
    replay = score(w,row,pad)
    if abs(replay['ce']-incumbent['ce']) > 1e-4:
        raise RuntimeError('Selected bias table cannot reproduce its recorded CE')
    record = dict(contract_sha256=h,sample_id=row['sample_id'],best=replay,
                  ordered_pairs=row['pairs'], groups_one_based=[[i+1 for i in g] for g in GROUPS],
                  causal_direction='receiver (later text) attends to sender (earlier text); reverse attention is never opened',
                  values=incumbent_values.tolist(),available_correct=bool(available),restarts=restarts)
    atomic_json(final,record)
    return record


def optimize_all(a,w,rows,singles,combined,pad,h,progress):
    cached = 0
    for row in rows:
        if (qdir(a,row)/'joint.json').exists():
            cached += a.joint_steps*2
        else:
            for restart in range(2):
                path=qdir(a,row)/f'optimization/start{restart}.pt'
                if path.exists(): cached += read_tensor(path,h)['step']
    progress.stage(5,len(rows)*2*a.joint_steps,initial=cached,unit='update',kind='train')
    output={}
    for row in rows:
        progress.names[5]=f'joint {row["dataset"]} 2x{a.joint_steps}'
        output[row['sample_id']]=optimize_question(a,w,row,singles[row['sample_id']],combined[row['sample_id']],pad,h,progress)
    progress.complete(f'All question-local optimizations completed; no shared predictor trained; {a.output_root}/questions')
    return output


def summarize(rows,singles,combined,joints,h):
    ids=[r['sample_id'] for r in rows]; base=[singles[i]['base'] for i in ids]; datasets=[r['dataset'] for r in rows]
    conditions={name:[singles[i]['summary'][key]['score'] for i in ids]
                for name,key in [('prior_sampled_single','sampled'),('all_pairs_single','exhaustive')]}
    conditions.update({f'raw_joint{k}':[next(x['score'] for x in combined[i] if x['k']==k) for i in ids] for k in (2,4,8)})
    conditions['optimized_joint']=[joints[i]['best'] for i in ids]
    metrics={name:compare_scores(base,values,datasets) for name,values in conditions.items()}
    for name,values in conditions.items():
        metrics[name]['mean_p_gold']=sum(x['p_gold'] for x in values)/len(rows)
        metrics[name]['baseline_mean_p_gold']=sum(x['p_gold'] for x in base)/len(rows)
        metrics[name]['delta_mean_p_gold_pp']=100*(metrics[name]['mean_p_gold']-metrics[name]['baseline_mean_p_gold'])
    details=[]
    for row in rows:
        i=row['sample_id']; single=singles[i]; joint=joints[i]
        hist=[x for restart in joint['restarts'] for x in restart['history']]
        details.append(dict(sample_id=i,dataset=row['dataset'],pairs=len(row['pairs']),baseline=single['base'],
            prior_backend_baseline=single['old'],single=single['summary'],joint=joint['best'],
            joint_available_correct=joint['available_correct'],
            nonzero_gradient_steps=sum(x['gradient_norm']>0 for x in hist),
            nonzero_update_steps=sum(x['update_norm']>0 for x in hist),
            max_gradient_norm=max((x['gradient_norm'] for x in hist),default=0),
            optimization_beats_measured_initialization=joint['best']['ce'] < initial_search(single,combined[i],len(row['pairs']))[0]['ce']-1e-4,
            compositions=[{k:v for k,v in c.items() if k not in ('values','contract_sha256')} for c in combined[i]]))
    metrics['optimized_joint']['available_correct']=sum(joints[i]['available_correct'] for i in ids)
    metrics['all_pairs_single']['available_correct']=sum(singles[i]['summary']['exhaustive']['available_correct'] for i in ids)
    newly=metrics['optimized_joint']['transitions'].get('W->C',0)
    # A genuinely insensitive question can have zero gradients. Do not mistake
    # that for a broken pipeline or gate away otherwise useful oracle effects.
    numerical_usable=any(x['nonzero_gradient_steps']>0 and x['nonzero_update_steps']>0 for x in details)
    screen=numerical_usable and newly>=4 and metrics['optimized_joint']['ce_reduction']>=.05
    diagnosis=dict(
        candidate_misses_observed=sum(not singles[i]['summary']['sampled']['available_correct'] and singles[i]['summary']['exhaustive']['available_correct'] for i in ids),
        joint_only_correct_found=sum(not singles[i]['summary']['exhaustive']['available_correct'] and joints[i]['available_correct'] for i in ids),
        raw_combination_harms_ce={str(k):sum(next(c for c in combined[i] if c['k']==k)['actual_gain'] < -.01 for i in ids) for k in (2,4,8)},
        gradient_and_update_check_passed=numerical_usable,
        questions_with_zero_gradients=sum(x['nonzero_gradient_steps']==0 for x in details),
        questions_with_zero_updates=sum(x['nonzero_update_steps']==0 for x in details),
        current_backend_baseline_correct=sum(x['correct'] for x in base),
        prior_backend_baseline_correct=sum(singles[i]['old']['correct'] for i in ids),
        baseline_prediction_drift=sum(singles[i]['old']['prediction']!=singles[i]['base']['prediction'] for i in ids))
    return dict(contract_sha256=h,status='complete',n=len(rows),missing=0,metrics=metrics,diagnosis=diagnosis,
        pilot_screen_passed=bool(screen),
        scope='Gold-conditioned search on 32 prior TRAIN errors. Conditional diagnostics, NOT representative/test/generalization accuracy.',
        next=('Only now investigate predicting useful interventions on separate TRAIN/dev data; no automatic scaling.' if screen else
              'Do not scale the predictor. Inspect measured candidate/joint/numerical diagnostics; bounded search failure is not an impossibility proof.'),
        limitations=['No KG-specific advantage is established without matched non-KG controls.',
                     'Exhaustive only for the stated finite single-action grid; joint search is local and budget-limited.',
                     'Fixed order, causal directions, 12 controlled layers, shared-head/group biases, bound 2 remain restrictions.',
                     'All metrics use the new matched numerical backend; previous numerical scores are drift diagnostics only.'],
        per_question=details)


def run(a,rows,old,contract,h,progress):
    mp=a.output_root/'manifest.json'
    if not mp.exists():
        git={k:subprocess.check_output(['git',*v],cwd=ROOT,text=True).strip() for k,v in
             [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current'])]}
        atomic_json(mp,dict(contract=contract,contract_sha256=h,git=git))
    atomic_jsonl(a.output_root/'cohort.jsonl',contract['cohort'])
    progress.complete(f'32 official-TRAIN errors; original Val/Test untouched; {mp}')
    final=a.output_root/'SUMMARY.json'
    if final.exists(): return read_json(final,h)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available():
        raise RuntimeError('Launch on the server with working GPU1; no model allocated')
    progress.stage(1,1)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error(); logging.disable_progress_bar()
    tok=AutoTokenizer.from_pretrained(contract['model'],local_files_only=True)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    llm=AutoModelForCausalLM.from_pretrained(contract['model'],local_files_only=True,
           dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    controller=OracleBiasTable(llm.config.hidden_size,rows[0]['pairs']).to('cuda')
    w=PropositionAttentionModel(llm,controller,attention_layers=tuple(i for g in GROUPS for i in g),
                               gradient_checkpointing=True,query_chunk=a.query_chunk).eval()
    install_oracle_attention()
    progress.complete('Frozen pristine Llama; no LoRA; only per-question bias tables are optimized')
    progress.stage(2,1)
    check=safety(w,max(rows,key=lambda r:len(r['input_ids'])),pad)
    check.update(contract_sha256=h,gpu=torch.cuda.get_device_name(0),peak_vram_bytes=torch.cuda.max_memory_allocated())
    atomic_json(a.output_root/'SAFETY.json',check)
    progress.weights[3]=check['calibration_forward_seconds']*sum(1+len(r['pairs'])*len(ACTIONS) for r in rows)
    progress.weights[4]=check['calibration_forward_seconds']*len(rows)*3
    progress.weights[5]=check['calibration_backward_seconds']*len(rows)*2*a.joint_steps
    progress.weights[6]=1.
    progress.complete('Grad/no-grad parity and longest-prompt backward checked; ETA calibrates forward and backward separately')
    singles=measure_singles(a,w,rows,old,pad,h,progress)
    combined=measure_combinations(a,w,rows,singles,pad,h,progress)
    joints=optimize_all(a,w,rows,singles,combined,pad,h,progress)
    progress.stage(6,1)
    report=summarize(rows,singles,combined,joints,h)
    atomic_json(final,report)
    atomic_jsonl(a.output_root/'per_question.jsonl',report['per_question'])
    progress.complete(f'All three diagnostics finished; {final}')
    return report


def main(argv=None):
    a=arguments(argv); torch.set_num_threads(8); torch.manual_seed(a.seed)
    if a.preflight_only:
        rows,old,contract,h=preflight(a)
        print(json.dumps(dict(status='preflight_ok',contract_sha256=h,questions=len(rows),
            datasets=dict(Counter(r['dataset'] for r in rows)),pairs=sum(len(r['pairs']) for r in rows),
            single_action_forwards=sum(len(r['pairs'])*len(ACTIONS) for r in rows),
            joint_optimizer_updates=len(rows)*2*a.joint_steps,max_tokens=max(len(r['input_ids']) for r in rows),
            output=str(a.output_root),model_loaded=False,gpu_job_started=False),indent=2)); return
    progress=Progress(STAGES)
    try:
        progress.stage(0,1)
        rows,old,contract,h=preflight(a)
        progress.log(f'[plan] 32 prior TRAIN errors -> frozen Llama -> safety/calibration -> all {sum(len(r["pairs"]) for r in rows):,} KG pairs x18 actions -> raw top2/4/8 -> 2x{a.joint_steps} updates/question -> diagnostic report. No performance gate skips later stages.')
        progress.log('[fairness] Same original text/order/choices; MATH backend on controlled layers for BOTH evaluation and gradients. Zero and old sampled actions are remeasured, not mixed with previous scores.')
        progress.log('[ETA] No compatible MATH+backward throughput yet. Longest-prompt safety calibrates both; fresh stages refine estimates. GPU1 only. Pair records and every 5 optimizer steps resume.')
        a.output_root.mkdir(parents=True,exist_ok=True)
        with (a.output_root/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=run(a,rows,old,contract,h,progress)
        if progress.index<6:
            for i in range(progress.index+1,7):
                progress.stage(i,1,initial=1);progress.complete('cached complete diagnostic')
        progress.log(f'[result] {result["status"]}; pilot screen={result["pilot_screen_passed"]}; {a.output_root}/SUMMARY.json; NOT test accuracy')
    except Exception as error:
        progress.log(f'[FAILED] stage={progress.index+1}/7 item={progress.done}/{progress.total}; {type(error).__name__}: {error}; durable={a.output_root}; identical command resumes completed pairs/optimizer checkpoints. No old results deleted.')
        raise
    finally: progress.close()


if __name__=='__main__': main()

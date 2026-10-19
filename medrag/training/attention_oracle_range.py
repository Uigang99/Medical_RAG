"""Paired range/time continuation from the SAME saved oracle bias table.

Scale-only outputs never seed optimization. Each bound gets fresh identical
Adam initialization. Report cumulative best-CE and raw endpoint scores apart.
This is gold-conditioned error recovery, not held-out QA accuracy.
"""
from __future__ import annotations

import torch

from medrag.training.attention_intervention_diagnostic import compare_scores
from medrag.training.proposition_pair_effects import paired_interval

SCALES = (.5, 1., 2., 4.)
BOUNDS = (2., 4., 8.)


def check_seed(row, artifact):
    if artifact['sample_id'] != row['sample_id'] or artifact['ordered_pairs'] != row['pairs']:
        raise RuntimeError('Saved oracle table does not match this question/pair order')
    values = torch.tensor(artifact['values'], dtype=torch.float32)
    if values.shape != (len(row['pairs']), 3) or not torch.isfinite(values).all():
        raise RuntimeError('Saved oracle table has invalid shape or values')
    if values.abs().max() > 2.000001:
        raise RuntimeError('Expected a saved oracle table bounded by 2')
    return values


def new_state(h, row, seed, bound):
    return dict(contract_sha256=h, sample_id=row['sample_id'], bound=float(bound), step=0,
                initial_values=seed['values'], best=seed['best'], best_values=seed['values'],
                best_step=0, available_correct=bool(seed['best']['correct']), history=[],
                milestones={}, complete=False)


def observe(state, result, values, step):
    state['available_correct'] |= result['correct']
    if result['ce'] < state['best']['ce']:
        state['best'], state['best_values'], state['best_step'] = result, values.tolist(), step


def milestone(state, endpoint, values, step):
    observe(state, endpoint, values, step)
    state['milestones'][str(step)] = dict(
        step=step, endpoint=endpoint, best=state['best'], best_values=state['best_values'],
        best_step=state['best_step'], available_correct=bool(state['available_correct']),
        endpoint_saturation_fraction=float((values.abs() >= state['bound']-1e-6).float().mean()),
        best_saturation_fraction=float((torch.tensor(state['best_values']).abs() >= state['bound']-1e-6).float().mean()))


def comparison(base, new, datasets):
    metric = compare_scores(base, new, datasets)
    for name in ('p_gold', 'margin'):
        before = sum(x[name] for x in base)/len(base)
        after = sum(x[name] for x in new)/len(new)
        scale = 100 if name == 'p_gold' else 1
        metric[name] = dict(baseline_mean=before, mean=after, delta=after-before,
                           delta_ci95=[scale*v for v in paired_interval([n[name]-b[name] for b,n in zip(base,new,strict=True)])],
                           ci_units='percentage_points' if name=='p_gold' else 'logit',
                           delta_pp=100*(after-before) if name=='p_gold' else None)
    return metric


def summarize(rows, seeds, scaling, optimized, steps, h):
    ids=[r['sample_id'] for r in rows]; datasets=[r['dataset'] for r in rows]
    baseline=[scaling[i]['zero'] for i in ids]
    incumbent=[seeds[i]['best'] for i in ids]
    values={f'scale_{scale:g}':[scaling[i]['scales'][f'{scale:g}'] for i in ids] for scale in SCALES}
    for bound in BOUNDS:
        for step in (steps//2, steps):
            values[f'bound_{bound:g}_step_{step}']=[optimized[i][bound]['milestones'][str(step)]['best'] for i in ids]
    conditions={name:dict(vs_no_bias=comparison(baseline,scores,datasets),
                         vs_saved_oracle=comparison(incumbent,scores,datasets)) for name,scores in values.items()}
    endpoints={}
    for bound in BOUNDS:
        for step in (steps//2,steps):
            measured=[optimized[i][bound]['milestones'][str(step)] for i in ids]
            endpoints[f'bound_{bound:g}_step_{step}']=dict(
                vs_no_bias=comparison(baseline,[m['endpoint'] for m in measured],datasets),
                vs_saved_oracle=comparison(incumbent,[m['endpoint'] for m in measured],datasets),
                endpoint_saturation_mean=sum(m['endpoint_saturation_fraction'] for m in measured)/len(ids),
                best_saturation_mean=sum(m['best_saturation_fraction'] for m in measured)/len(ids),
                any_correct_seen=sum(m['available_correct'] for m in measured))
    contrasts={}
    for scale in (.5,2.,4.):
        contrasts[f'scale_{scale:g}_vs_1']=comparison(incumbent,values[f'scale_{scale:g}'],datasets)
    for bound in BOUNDS:
        contrasts[f'bound_{bound:g}_time_{steps//2}_to_{steps}']=comparison(
            values[f'bound_{bound:g}_step_{steps//2}'],values[f'bound_{bound:g}_step_{steps}'],datasets)
    for bound in (4.,8.):
        for step in (steps//2,steps):
            contrasts[f'bound_{bound:g}_vs_2_step_{step}']=comparison(
                values[f'bound_2_step_{step}'],values[f'bound_{bound:g}_step_{step}'],datasets)
    detail=[]
    for row in rows:
        i=row['sample_id']
        detail.append(dict(sample_id=i,dataset=row['dataset'],pairs=len(row['pairs']),zero=scaling[i]['zero'],
            saved_oracle=seeds[i]['best'],scaled=scaling[i]['scales'],
            optimization={f'{b:g}':dict(milestones={s:{k:v for k,v in m.items() if k!='best_values'}
                            for s,m in optimized[i][b]['milestones'].items()},
                nonzero_gradient_steps=sum(v['gradient_norm']>0 for v in optimized[i][b]['history']),
                nonzero_update_steps=sum(v['update_norm']>0 for v in optimized[i][b]['history'])) for b in BOUNDS}))
    screens={f'{b:g}':bool(conditions[f'bound_{b:g}_step_{steps}']['vs_no_bias']['correct']>=4 and
                    conditions[f'bound_{b:g}_step_{steps}']['vs_no_bias']['ce_reduction']>=.05) for b in BOUNDS}
    return dict(contract_sha256=h,status='complete',n=len(rows),missing=0,conditions=conditions,
        raw_endpoints=endpoints,
        paired_contrasts=contrasts,capacity_screen_by_bound=screens,
        scope='Gold-conditioned, error-selected official TRAIN32. Not test accuracy or generalization; conditional bootstrap intervals only.',
        interpretation=dict(
            scale='Same pattern, different amplitude. Raw scaling scores are not replaced by a no-op oracle.',
            time='Bound 2 continuation isolates extra search from a larger allowed range.',
            range='Compare bounds at the SAME update count and identical seed/optimizer initialization.',
            no_adaptive_extension='Stop at fixed bounds 2/4/8 and configured steps regardless of outcome.'),
        limitations=['Only the saved best table is used as initialization; this does not prove a global optimum.',
                     'Causal directions, text order, layer groups and head sharing remain unchanged.',
                     'Best-CE selection is gold-conditioned; an observed ability to fix errors is not a deployable policy.'],
        per_question=detail)

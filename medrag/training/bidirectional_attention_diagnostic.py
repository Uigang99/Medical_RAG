"""Reporting for matched causal/bidirectional, zero-bias/oracle comparisons."""
from __future__ import annotations

from medrag.training.attention_oracle_range import comparison


def summarize(rows, records, steps, h):
    ds=[r['dataset'] for r in rows]
    values={}
    for mode in ('causal','bidirectional'):
        values[mode+'_zero']=[records[r['sample_id']][mode]['initial'] for r in rows]
        for step in (steps//2,steps):
            values[f'{mode}_oracle_{step}']=[records[r['sample_id']][mode]['milestones'][str(step)]['best'] for r in rows]
            values[f'{mode}_endpoint_{step}']=[records[r['sample_id']][mode]['milestones'][str(step)]['endpoint'] for r in rows]
    baseline=values['causal_zero']
    conditions={k:comparison(baseline,v,ds) for k,v in values.items()}
    contrasts={
        'opening_only':comparison(baseline,values['bidirectional_zero'],ds),
        'causal_optimization':comparison(baseline,values[f'causal_oracle_{steps}'],ds),
        'bidirectional_optimization':comparison(values['bidirectional_zero'],values[f'bidirectional_oracle_{steps}'],ds),
        'matched_oracle_direction':comparison(values[f'causal_oracle_{steps}'],values[f'bidirectional_oracle_{steps}'],ds),
        'matched_raw_endpoint_direction':comparison(values[f'causal_endpoint_{steps}'],values[f'bidirectional_endpoint_{steps}'],ds),
    }
    subgroups={}
    for label,predicate in [('initially_correct',lambda i:baseline[i]['correct']),
                            ('initially_wrong',lambda i:not baseline[i]['correct']),
                            ('graphless',lambda i:not rows[i]['pairs'])]:
        ix=[i for i in range(len(rows)) if predicate(i)]
        subgroups[label]={k:comparison([baseline[i] for i in ix],[v[i] for i in ix],[ds[i] for i in ix])
                          for k,v in values.items()} if ix else {'n':0}
    details=[]
    for i,row in enumerate(rows):
        details.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],pairs=len(row['pairs']),
            source_initially_correct=records[row['sample_id']]['causal']['source_initially_correct'],
            scores={k:v[i] for k,v in values.items()},
            optimization={mode:dict(best_step=records[row['sample_id']][mode]['best_step'],
                any_correct_seen=records[row['sample_id']][mode]['available_correct'],
                steps_executed=records[row['sample_id']][mode]['updates'],
                forward_gradient_nonzero_steps=sum(x['forward_gradient_norm']>0 for x in records[row['sample_id']][mode]['history']),
                reverse_gradient_nonzero_steps=sum(x['reverse_gradient_norm']>0 for x in records[row['sample_id']][mode]['history']))
                for mode in ('causal','bidirectional')}))
    primary=contrasts['matched_oracle_direction']
    net=primary['correct']-primary['baseline_correct']
    screen=bool(net>=3 and primary['ce_reduction']>=.02)
    return dict(status='complete',contract_sha256=h,n=len(rows),missing=0,conditions=conditions,
        contrasts=contrasts,subgroups=subgroups,per_question=details,
        primary='bidirectional versus causal best-CE oracle at matched steps, same zero initialization',
        minimum_worthwhile=dict(net_extra_correct=3,mean_ce_reduction=.02),screen_passed=screen,
        baseline_drift_from_source=sum(b['correct']!=records[r['sample_id']]['causal']['source_initially_correct'] for r,b in zip(rows,baseline,strict=True)),
        limitations=[
            'Reused official TRAIN128, not held-out or final test. Oracle uses gold per question; no deployable controller.',
            'Best-CE is selected within EACH mask condition including its own zero. No cross-mask fallback hides damage.',
            'Opening future KG blocks changes computation; bidirectional oracle also has twice the directional parameters.',
            'Source proposition order and subject/object role metadata are unchanged; relation meanings do not select a direction.',
            'Only the original 12 controlled layers are modified; other layers and within-proposition attention remain causal.',
            'This is constrained local search, not a global optimum. Failure does not rule out adapted models.',
            'Bootstrap intervals are conditional exploratory summaries, not confirmatory estimates of test improvement.'])

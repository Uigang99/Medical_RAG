"""Execution-only speedups; no changes to masks, CE, optimizer, or update budget.

Cache the immutable lower-layer output; trade activation memory for avoiding
upper-layer recomputation. Retain the exact query chunk and MATH backend.
Only question-local Oracle scalars/Adam state are rolled back after a real OOM.
"""
from __future__ import annotations

import gc
import os
import statistics
import time
import torch

from medrag.training.attention_oracle import optimize_step, score


def deterministic_execution():
    """Require reproducible GPU reductions before comparing Adam trajectories."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=False)


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def step_with_retry(w,row,pad,opt,bound,log=None):
    """Keep exactly one Adam update, even if the fast attempt OOMs midway."""
    if w.gradient_checkpointing:
        return optimize_step(w,row,pad,opt,bound)
    values=cpu_copy(w.controller.values)
    optimizer=cpu_copy(opt.state_dict())
    failed=False
    try:
        result=optimize_step(w,row,pad,opt,bound)
    except torch.cuda.OutOfMemoryError as exc:
        # Release the failed autograd graph before allocating retry activations.
        exc.__traceback__=None
        failed=True
    if not failed:return result
    opt.zero_grad(set_to_none=True)
    w.controller.assign(values);opt.load_state_dict(optimizer)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    w.gradient_checkpointing=True
    if log:log(f'[memory] {row["sample_id"]}: OOM; exact Adam step rolled back; checkpointing for this question only')
    result,before,diagnostic=optimize_step(w,row,pad,opt,bound)
    diagnostic['oom_retried_with_checkpointing']=True
    return result,before,diagnostic


def probe(w,row,pad,cache,checkpointing):
    """Three throwaway updates; no probe table seeds the actual Oracle search."""
    w.cache_prefix=cache;w.gradient_checkpointing=checkpointing
    w.controller.assign(torch.full_like(w.controller.values,.25))
    opt=torch.optim.Adam([w.controller.values],lr=.1)
    # Warm up prefix, kernels, allocator; timings exclude one-off initialization.
    score(w,row,pad)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    times=[];first=None
    for _ in range(3):
        start=time.monotonic()
        result,_,_=optimize_step(w,row,pad,opt,8.)
        torch.cuda.synchronize();times.append(time.monotonic()-start)
        if first is None:first=dict(score=result,gradient=cpu_copy(w.controller.values.grad))
    start=time.monotonic();evaluation=score(w,row,pad);torch.cuda.synchronize()
    return dict(cache_prefix=cache,gradient_checkpointing=checkpointing,
        update_seconds=statistics.median(times[1:]),forward_seconds=time.monotonic()-start,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),first=first,
        endpoint=evaluation,values=cpu_copy(w.controller.values))


def assert_equivalent(actual,reference):
    # No relaxed BF16 tolerance to hide an altered execution path.
    torch.testing.assert_close(torch.tensor(actual['first']['score']['logits']),
        torch.tensor(reference['first']['score']['logits']),atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(torch.tensor(actual['endpoint']['logits']),
        torch.tensor(reference['endpoint']['logits']),atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(actual['first']['gradient'],reference['first']['gradient'],atol=1e-7,rtol=1e-4)
    torch.testing.assert_close(actual['values'],reference['values'],atol=1e-5,rtol=1e-4)


def calibrate(w,row,pad,memory_fraction,log,save_report=None):
    """Longest prompt, both masks: pick the faster numerically equivalent mode.

    The cache is required to match the reference; memory-vs-recompute selection
    may fall back to checkpointing. Non-OOM runtime failures are not swallowed.
    """
    if not 0.5<=memory_fraction<=0.92:raise ValueError('memory_fraction must be in [0.5,0.92]')
    w.runtime_profiles={}
    report=dict(status='calibrating',sample_id=row['sample_id'],tokens=len(row['input_ids']),modes={},
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        memory_fraction=memory_fraction,query_chunk=w.query_chunk,
        selection='Same MATH/query shape. Compare loss/logits, bias gradients and 3-step Adam values; select by time, not accuracy.',
        note='Longest-row warm-step estimate, not measured full-cohort wall time. OOM recovery is question-local.')
    for mode in ('causal','bidirectional'):
        w.configure(row,mode=='bidirectional')
        reference=probe(w,row,pad,False,True)
        attempts=[];valid=[]
        report['modes'][mode]=dict(attempts=attempts,old_update_seconds=reference['update_seconds'],
            old_peak_vram_bytes=reference['peak_vram_bytes'])
        # A nondeterministic baseline cannot judge whether the optimization is equivalent.
        repeat=probe(w,row,pad,False,True)
        try:
            assert_equivalent(repeat,reference)
        except AssertionError as exc:
            report.update(status='failed',failure='baseline_not_repeatable',reason=str(exc))
            if save_report:save_report(report)
            raise RuntimeError(f'{mode} baseline fails its own repeatability test; this is not a VRAM shortage: {exc}') from exc
        report['modes'][mode]['baseline_repeat_parity']=True
        del repeat
        for ckpt in (False,True):
            opt_result=None;oom=False
            w.controller.values.grad=None;gc.collect();torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info()
            allocated=torch.cuda.memory_allocated()
            limit=min(total*memory_fraction,allocated+free*memory_fraction)
            try:
                opt_result=probe(w,row,pad,True,ckpt)
            except torch.cuda.OutOfMemoryError as exc:
                exc.__traceback__=None;oom=True
            if oom:
                w.controller.values.grad=None;gc.collect();torch.cuda.empty_cache()
                attempts.append(dict(gradient_checkpointing=ckpt,status='oom'))
                if save_report:save_report(report)
                log(f'[calibration] {mode}: retain activations OOM; testing checkpointed path')
                continue
            try:
                assert_equivalent(opt_result,reference)
            except AssertionError as exc:
                attempts.append(dict(gradient_checkpointing=ckpt,status='parity_failed',reason=str(exc)))
                if save_report:save_report(report)
                log(f'[calibration] {mode}: checkpointing={ckpt} failed strict parity; path rejected: {exc}')
                continue
            item={k:v for k,v in opt_result.items() if k not in ('first','values','endpoint')}
            item.update(status='safe' if item['peak_vram_bytes']<=limit else 'insufficient_headroom',
                vram_limit_bytes=int(limit),parity=True)
            attempts.append(item)
            if save_report:save_report(report)
            if item['status']=='safe':valid.append(item)
        if not valid:
            reasons=[x['status'] for x in attempts]
            report.update(status='failed',failure='no_valid_runtime',reasons=reasons)
            if save_report:save_report(report)
            raise RuntimeError(f'No valid {mode} runtime: {reasons}. '
                'parity_failed means numerical mismatch, NOT insufficient GPU memory; inspect RUNTIME_CALIBRATION.json')
        selected=min(valid,key=lambda x:x['update_seconds'])
        report['modes'][mode].update(selected=selected,
            measured_warm_step_speedup=reference['update_seconds']/selected['update_seconds'])
        log(f'[calibration] {mode}: checkpointing={selected["gradient_checkpointing"]}, '
            f'{selected["update_seconds"]:.3f} s/update, peak={selected["peak_vram_bytes"]/1024**3:.1f} GiB; '
            f'matched old={reference["update_seconds"]:.3f} s/update')
        del reference
    w.cache_prefix=True
    w.runtime_profiles={mode:data['selected'] for mode,data in report['modes'].items()}
    w.controller.values.grad=None;gc.collect();torch.cuda.empty_cache()
    report['status']='complete'
    if save_report:save_report(report)
    return report

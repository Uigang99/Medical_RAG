#!/usr/bin/env python3
"""Add matched No-RAG/Base-RAG CoT to the completed PCED run; Macro-3 primary."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import evaluate_question_only_pced as reference
from medrag.generation.question_only_cot import ARMS, BatchedCoT, render_prompt
from medrag.generation.question_only_pced import RATIONALE_SYSTEM, kv_bytes_per_question
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, jsonl, sha_file
from medrag.training.graph_input_progress import Progress

VERSION = 'question_only_cot_baselines_pced_matched_v1'
GROUPS = ('medmcqa', 'medqa', 'mmlu')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pced-root', type=Path, default=ROOT/'results/question_only_pced_medcpt_two_modes_v1')
    p.add_argument('--output-root', type=Path, default=ROOT/'results'/VERSION)
    p.add_argument('--max-batch-questions', type=int, default=64)
    p.add_argument('--memory-fraction', type=float, default=.88)
    p.add_argument('--limit', type=int, default=0, help='0=all 6545; otherwise frozen hash-selected smoke cohort')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if a.max_batch_questions < 1 or not 0 < a.memory_fraction < .98 or a.limit < 0:
        p.error('Invalid runtime/cohort arguments')
    a.pced_root, a.output_root = a.pced_root.resolve(), a.output_root.resolve()
    if a.limit and a.output_root == (ROOT/'results'/VERSION).resolve():
        a.output_root = a.output_root.with_name(VERSION+f'_smoke{a.limit}')
    if (a.output_root == a.pced_root or a.pced_root in a.output_root.parents
            or a.output_root in a.pced_root.parents):
        p.error('New outputs must be separate from the immutable PCED result directory')
    return a


def group(dataset):
    return 'mmlu' if dataset.startswith('mmlu_') else dataset


def verify_reference(a):
    manifest = json.loads((a.pced_root/'manifest.json').read_text())
    c = manifest['contract']
    if digest(c) != manifest['contract_sha256'] or c['version'] != reference.VERSION:
        raise RuntimeError('Unsupported/corrupt reference manifest')
    if c['top_k'] != 8 or c['systems'][1] != RATIONALE_SYSTEM or len(c['sample_ids']) != 6545:
        raise RuntimeError('Expected the completed full Top-8 PCED experiment')
    g = c['generation']
    if (g['greedy'] is not True or g['dtype'] != 'bfloat16' or g['attention'] != 'sdpa'
            or g['max_rationale_tokens'] != 512 or g['max_model_len'] != 8192):
        raise RuntimeError('Unsupported reference decoding settings; never silently substitute')
    for path, expected in c['inputs_sha256'].items():
        if sha_file(path) != expected:
            raise RuntimeError(f'Reference input changed: {path}')
    for path, expected in c['code_sha256'].items():
        if sha_file(ROOT/path) != expected:
            raise RuntimeError(f'Reference implementation changed: {path}')
    for name, expected in c['weights'].items():
        stat = (Path(c['model'])/name).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise RuntimeError(f'Reference weights changed: {name}')
    for name, version in c['runtime']['packages'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'{name} version differs from PCED; use the same environment')
    caches = [Path(p) for p in c['inputs_sha256'] if p.endswith('/balanced_ce.jsonl')]
    tests = [Path(p) for p in c['inputs_sha256'] if p.endswith('/medmcqa/test.jsonl')]
    if len(caches) != 1 or len(tests) != 1:
        raise RuntimeError('Ambiguous reference inputs')
    cfg = SimpleNamespace(model=Path(c['model']), retrieval_cache=caches[0], benchmark_root=tests[0].parent.parent,
                          max_model_len=g['max_model_len'], max_rationale_tokens=g['max_rationale_tokens'], limit=a.limit)
    return manifest, cfg


def prepare(a, cfg, manifest, tokenizer):
    rows, gold, labels, counts = reference.prepare(cfg, tokenizer)
    if labels != manifest['contract']['label_ids']:
        raise RuntimeError('Label token IDs differ from PCED')
    found_groups = {group(r['dataset']) for r in rows}
    if found_groups != set(GROUPS):
        raise RuntimeError('Macro-3 needs all three groups; increase --limit for a smoke run')
    paired = {r['sample_id']: r for r in jsonl(a.pced_root/'paired_predictions.jsonl')}
    expected = set(manifest['contract']['sample_ids'])
    if len(paired) != len(expected) or set(paired) != expected:
        raise RuntimeError('Incomplete reference paired predictions')
    records = reference.load_done(SimpleNamespace(output_root=a.pced_root), 'rationale', rows, manifest['contract_sha256'])
    if len(records) != len(rows):
        raise RuntimeError('PCED rationale reference is incomplete; no GPU work started')
    for r in rows:
        sid = r['sample_id']; saved = records[sid]
        if (paired[sid]['gold'] != gold[sid] or paired[sid]['rationale'] != saved['answer']
                or saved['dataset'] != r['dataset'] or saved['priors'] != r['priors']
                or saved['document_ids'] != [d['corpus_id'] for d in r['documents']]):
            raise RuntimeError(f'Reference record alignment failed: {sid}')
        r['baseline_lengths'] = {}
        for arm in ARMS:
            prompt = render_prompt(tokenizer, r, arm)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            reserve = cfg.max_rationale_tokens + len(tokenizer.encode('\nFinal answer:', add_special_tokens=False)) + 1
            if len(ids)+reserve > cfg.max_model_len:
                raise RuntimeError(f'{arm} context overflow: {sid}; no truncation or dropped questions allowed')
            prefix = tokenizer.encode(prompt+'\nFinal answer:', add_special_tokens=False)
            for key, label_id in zip('ABCD', labels):
                full = tokenizer.encode(prompt+'\nFinal answer: '+key, add_special_tokens=False)
                if full != prefix+[label_id]:
                    raise RuntimeError(f'Unstable label token: {sid}/{arm}/{key}')
            r['baseline_lengths'][arm] = len(ids)
    return rows, gold, labels, counts, records


def contract(a, ref, rows, records):
    paths = [Path(__file__), ROOT/'medrag/generation/question_only_cot.py', Path(reference.__file__)]
    return {'version': VERSION, 'hypothesis': 'Does PCED outperform ordinary RAG when all methods generate CoT?',
            'reference_contract_sha256': ref['contract_sha256'], 'reference_root': str(a.pced_root),
            'reference_predictions_sha256': digest([records[r['sample_id']] for r in rows]),
            'code_sha256': {str(p.relative_to(ROOT)): sha_file(p) for p in paths},
            'git_at_creation': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'model': ref['contract']['model'], 'weights': ref['contract']['weights'],
            'sample_ids': [r['sample_id'] for r in rows], 'split': 'frozen TEST; no selection or tuning',
            'arms': [*ARMS,'pced_cot'], 'system': RATIONALE_SYSTEM,
            'generation': {k: v for k,v in ref['contract']['generation'].items() if k not in ('answer','history')},
            'answer': 'one space-prefixed A/B/C/D token, selected by ordinary logits for baselines, PCED scores for PCED',
            'history': 'each baseline retains its own generated tokens and KV cache; PCED retains its existing shared-token nine-stream protocol',
            'generation_note': 'Baselines use ordinary greedy single-stream logits, including A/B/C/D selection; ONLY PCED uses expert contrast/prior/max. Same marker/EOS/budget/cache behavior.',
            'input': 'question/options then no evidence or all exact Top-8 chunks in CE order; rank, optional title, text',
            'primary': 'Macro-3 PCED-CoT minus Base-RAG-CoT; pooled MMLU accuracy, not macro over six subjects',
            'secondary': 'Macro-3 other contrasts, micro/dataset accuracy, paired transitions, stratified paired bootstrap 95% CI, tokens/termination/time',
            'bootstrap': {'seed':42,'resamples':5000,'strata':list(GROUPS)},
            'runtime': {'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'max_batch_questions':a.max_batch_questions,
                        'memory_fraction':a.memory_fraction,'packages':ref['contract']['runtime']['packages']},
            'fairness': 'Same model, source hashes, system, backend/dtype, context limit, output protocol and cohort. Inputs/decoding intentionally differ by method; batch shapes need not be bit-identical.'}


def validate_record(row, saved, arm, h):
    answer = saved.get('answer')
    docs = [d['corpus_id'] for d in row['documents']] if arm == 'base_rag_cot' else []
    if (saved.get('sample_id') != row['sample_id'] or saved.get('dataset') != row['dataset']
            or saved.get('arm') != arm or saved.get('contract_sha256') != h
            or answer not in ('A','B','C','D') or saved.get('document_ids') != docs
            or saved.get('finish_reason') not in ('natural_marker','eos','token_budget')
            or not isinstance(saved.get('rationale_token_ids'),list)):
        raise RuntimeError(f'Invalid/resume-incompatible baseline result: {row["sample_id"]}/{arm}')


def load_done(a, rows, arm, h):
    expected = {r['sample_id']:r for r in rows}; done = {}
    for p in sorted((a.output_root/arm/'shards').glob('*.jsonl')):
        for saved in jsonl(p):
            sid = saved.get('sample_id')
            if sid not in expected or sid in done:
                raise RuntimeError(f'Duplicate/unexpected cached question: {p}/{sid}')
            validate_record(expected[sid],saved,arm,h)
            done[sid] = saved
    return done


def summarize(rows, gold, results, resamples=5000):
    import numpy as np
    names = [*ARMS,'pced_cot']; n = len(rows)
    if not n or any(set(results[m]) != {r['sample_id'] for r in rows} for m in names):
        raise RuntimeError('Cannot summarize incomplete/mismatched results')
    flags = np.array([[results[m][r['sample_id']]['answer']==gold[r['sample_id']] for m in names] for r in rows],dtype=float)
    groups = {g: np.array([i for i,r in enumerate(rows) if group(r['dataset'])==g],dtype=int) for g in GROUPS}
    if any(not len(ix) for ix in groups.values()):
        raise RuntimeError('Macro-3 requires all three groups')
    means = np.array([flags[ix].mean(0) for ix in groups.values()]).mean(0)
    rng = np.random.default_rng(42)
    boot = np.array([np.array([flags[ix[rng.integers(len(ix),size=len(ix))]].mean(0)
                               for ix in groups.values()]).mean(0) for _ in range(resamples)])
    conditions = {}
    for j,name in enumerate(names):
        by_group = {g:{'n':len(ix),'correct':int(flags[ix,j].sum()),'accuracy':float(flags[ix,j].mean())} for g,ix in groups.items()}
        by_dataset = {}
        for d in sorted({r['dataset'] for r in rows}):
            ix = [i for i,r in enumerate(rows) if r['dataset']==d]
            by_dataset[d] = {'n':len(ix),'correct':int(flags[ix,j].sum()),'accuracy':float(flags[ix,j].mean())}
        values = list(results[name].values())
        conditions[name] = {'n':n,'correct':int(flags[:,j].sum()),'macro3_accuracy':float(means[j]),
            'micro_accuracy':float(flags[:,j].mean()),'by_group':by_group,'by_dataset':by_dataset,
            'finish_reasons':dict(Counter(v['finish_reason'] for v in values)),
            'mean_rationale_tokens':sum(len(v['rationale_token_ids']) for v in values)/n}
    contrasts = {}
    for left,right in [(0,1),(0,2),(1,2)]:
        a,b=flags[:,left].astype(bool),flags[:,right].astype(bool)
        transitions={f'{x}->{y}':int(((a==(x=='C'))&(b==(y=='C'))).sum()) for x in ('C','W') for y in ('C','W')}
        contrasts[f'{names[right]}_minus_{names[left]}'] = {
            'baseline':names[left],'comparison':names[right],'n':n,
            'macro3_delta':float(means[right]-means[left]),
            'macro3_delta_95_ci':np.quantile(boot[:,right]-boot[:,left],[.025,.975]).tolist(),
            'micro_delta':float((flags[:,right]-flags[:,left]).mean()),'transitions':transitions,
            'transition_rates_all_questions':{k:v/n for k,v in transitions.items()},
            'macro3_recovery_rate':float(np.mean([((~a[ix])&b[ix]).mean() for ix in groups.values()])),
            'macro3_harm_rate':float(np.mean([(a[ix]&(~b[ix])).mean() for ix in groups.values()]))}
    return {'version':VERSION,'n':n,'missing':0,'primary_metric':'macro3_accuracy',
            'primary_contrast':'pced_cot_minus_base_rag_cot','macro3_definition':'mean(MedMCQA accuracy, MedQA accuracy, pooled medical MMLU accuracy)',
            'conditions':conditions,'paired':contrasts,'bootstrap_resamples':resamples,
            'interpretation':'All three arms use CoT and matched inference protocol. Raw transition counts are micro statistics, not the Macro-3 primary effect.'}


def run(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error()
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=Progress(['CPU matched preflight','load frozen Llama','No-RAG CoT TEST','Base-RAG CoT TEST','Macro-3 comparison'])
        progress.weights[-1]=2.0
        stage='CPU preflight'; completed={}
        try:
            progress.stage(0,1)
            ref,cfg=verify_reference(a)
            tokenizer=AutoTokenizer.from_pretrained(cfg.model,local_files_only=True,padding_side='left')
            if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
            rows,gold,labels,counts,pced=prepare(a,cfg,ref,tokenizer)
            c=contract(a,ref,rows,pced); manifest_path=a.output_root/'manifest.json'
            if manifest_path.exists():c['git_at_creation']=json.loads(manifest_path.read_text())['contract']['git_at_creation']
            h=ensure_contract(manifest_path,c)
            results={arm:load_done(a,rows,arm,h) for arm in ARMS};results['pced_cot']=pced
            completed={arm:len(v) for arm,v in results.items()}
            atomic_json(a.output_root/'preflight.json',{'n':len(rows),'datasets':counts,'label_ids':labels,
                'reference_complete':True,'max_prompt_tokens':{arm:max(r['baseline_lengths'][arm] for r in rows) for arm in ARMS}})
            progress.complete(f'N={len(rows)}; cached={completed}; reference unchanged; {a.output_root}')
            progress.log('[plan] fixed TEST+PCED contract -> one frozen Llama -> No-RAG CoT -> Top-8 Base-RAG CoT -> Macro-3; no retrieval/training; GPU1')
            progress.log('[ETA] New single-stream rates unknown. Each baseline calibrates separately; completed PCED is reused, not timed as fresh work. Batched JSONL saves replace per-question fsync.')
            if a.preflight_only:
                progress.log('[preflight passed] complete cohort, original PCED hashes/results, exact Top-8, prompt lengths and labels validated; no GPU/model loading')
                return
            stage='load frozen Llama';progress.stage(1,1)
            if any(len(results[arm])<len(rows) for arm in ARMS):
                if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU --preflight-only is supported')
                torch.manual_seed(42);random.seed(42)
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
                model=AutoModelForCausalLM.from_pretrained(cfg.model,local_files_only=True,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
                model.generation_config.pad_token_id=tokenizer.pad_token_id
                engine=BatchedCoT(model,tokenizer,labels,max_rationale_tokens=cfg.max_rationale_tokens)
            progress.complete('same model/backend/dtype as PCED; one stream per baseline question')
            rp=a.output_root/'runtime.json';runtime=json.loads(rp.read_text()) if rp.exists() else {}
            for i,arm in enumerate(ARMS,2):
                if runtime.get(arm,{}).get('seconds_per_question'):
                    progress.weights[i]=runtime[arm]['seconds_per_question']*len(rows)
            for i,arm in enumerate(ARMS,2):
                stage=arm;progress.stage(i,len(rows),initial=len(results[arm]),unit='q',kind='generation')
                pending=sorted((r for r in rows if r['sample_id'] not in results[arm]),key=lambda r:(r['baseline_lengths'][arm],r['sample_id']))
                offset=0;cap=a.max_batch_questions;fresh=0;elapsed=0.0
                if pending:torch.cuda.reset_peak_memory_stats()
                while offset<len(pending):
                    window=pending[offset:offset+cap];free,total=torch.cuda.mem_get_info()
                    available=max(0,min(free,total*a.memory_fraction-torch.cuda.memory_allocated())-6*1024**3)
                    kv=kv_bytes_per_question(model.config,max(r['baseline_lengths'][arm] for r in window),cfg.max_rationale_tokens+16,streams=1)
                    count=min(len(window),max(1,int(available*.5/max(1,kv))));batch=window[:count]
                    stage=f'{arm}; first={batch[0]["sample_id"]}; completed={len(results[arm])}/{len(rows)}'
                    start=time.monotonic()
                    try:generated=engine.generate_plain(batch,arm)
                    except torch.OutOfMemoryError:
                        gc.collect();torch.cuda.empty_cache()
                        if count==1:raise
                        cap=max(1,count//2);progress.log(f'[OOM retry] {arm} uncommitted batch {count}->{cap}; same inputs/budget');continue
                    if {v['sample_id'] for v in generated}!={r['sample_id'] for r in batch} or len(generated)!=len(batch):
                        raise RuntimeError('Decoder returned incomplete/duplicate batch')
                    for v in generated:v['contract_sha256']=h
                    path=a.output_root/arm/'shards'/('batch_'+hashlib.sha256(batch[0]['sample_id'].encode()).hexdigest()+'.jsonl')
                    if path.exists():raise RuntimeError(f'Refusing to overwrite existing batch: {path}')
                    atomic_jsonl(path,generated)
                    for v in generated:results[arm][v['sample_id']]=v
                    elapsed+=time.monotonic()-start;fresh+=count;offset+=count;completed[arm]=len(results[arm])
                    progress.update(count)
                    runtime[arm]={'fresh_questions':fresh,'seconds_including_batch_save':elapsed,'seconds_per_question':elapsed/fresh,
                        'last_batch_questions':count,'gpu':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                    atomic_json(rp,runtime)
                progress.complete(f'{len(results[arm])}/{len(rows)}; {a.output_root/arm}')
            stage='Macro-3 comparison';progress.stage(4,1)
            report=summarize(rows,gold,results)
            atomic_json(a.output_root/'summary.json',report)
            atomic_jsonl(a.output_root/'paired_predictions.jsonl',({'sample_id':r['sample_id'],'dataset':r['dataset'],'gold':gold[r['sample_id']],
                **{arm:results[arm][r['sample_id']]['answer'] for arm in results}} for r in rows))
            progress.complete(str(a.output_root/'summary.json'))
            progress.log('[complete] Macro-3 '+json.dumps({m:round(v['macro3_accuracy']*100,3) for m,v in report['conditions'].items()})+'; PCED reference unchanged')
        except Exception as exc:
            progress.log(f'[FAILED] stage={stage}; completed={completed}; durable={a.output_root}; {type(exc).__name__}: {exc}; identical command resumes committed batches, reference unchanged')
            raise
        finally:progress.close()


if __name__=='__main__':
    run(arguments())

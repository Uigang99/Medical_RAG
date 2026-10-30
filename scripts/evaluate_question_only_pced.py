#!/usr/bin/env python3
"""Frozen question-only Top-8 PCED: paired direct-label and shared-token CoT.

Reuses the verified benchmark loader, exact source-text prompt format, atomic
writers, contract validation and progress footer of the question-only baseline.
No retrieval, training, answer-conditioned routing or TEST tuning is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

from medrag.benchmark import load_benchmark_samples
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, ensure_contract, jsonl, sha_file
from medrag.training.graph_input_progress import Progress
from medrag.generation.question_only_pced import (
    BatchedPCED, DIRECT_SYSTEM, RATIONALE_SYSTEM, medcpt_priors, render_prompt, kv_bytes_per_question)

VERSION = 'question_only_pced_medcpt_two_modes_v1'
MODES = ('direct', 'rationale')
DATASETS = ('medmcqa', 'medqa', 'mmlu_anatomy', 'mmlu_clinical_knowledge',
            'mmlu_college_biology', 'mmlu_college_medicine', 'mmlu_medical_genetics',
            'mmlu_professional_medicine')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval-cache', type=Path, default=ROOT / 'databases/run_cache/question_only_ip_fourway_test6545_v1/balanced_ce.jsonl')
    p.add_argument('--benchmark-root', type=Path, default=ROOT / 'datasets/benchmark/mcq/unified')
    p.add_argument('--model', type=Path, default=ROOT.parent / 'models/Llama-3-8B-Instruct')
    p.add_argument('--output-root', type=Path, default=ROOT / 'results' / VERSION)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--max-rationale-tokens', type=int, default=512)
    p.add_argument('--gamma', type=float, default=2.5)
    p.add_argument('--max-batch-questions', type=int, default=32)
    p.add_argument('--memory-fraction', type=float, default=.88)
    p.add_argument('--limit', type=int, default=0, help='0: all 6545; >0: fixed hash-selected smoke cohort')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if (not 0 < a.memory_fraction < .98 or a.max_batch_questions < 1 or a.limit < 0
            or a.max_rationale_tokens < 1 or not math.isfinite(a.gamma) or a.gamma < 0):
        p.error('Invalid memory, batch, cohort, token budget or gamma')
    for key in ('retrieval_cache', 'benchmark_root', 'model', 'output_root'):
        setattr(a, key, getattr(a, key).resolve())
    if a.limit and a.output_root == (ROOT / 'results' / VERSION).resolve():
        a.output_root = a.output_root.with_name(VERSION + f'_smoke{a.limit}')
    return a


def prepare(a, tokenizer):
    # Same loader/cohort and space-prefixed label convention as the existing
    # evaluate_question_only_ce_mcq_baselines.py; no dependency on that local script.
    samples = [s for d in DATASETS for s in load_benchmark_samples(
        a.benchmark_root/d/'test.jsonl', 'mcq', 'unified', d, 'test')]
    if len(samples) != 6545 or len({s.id for s in samples}) != 6545:
        raise RuntimeError('Expected 6545 distinct TEST questions')
    selected = {s.id for s in samples}
    if a.limit:
        selected = set(sorted(selected, key=lambda x: hashlib.sha256(('42:'+x).encode()).hexdigest())[:a.limit])
    rows, gold, counts = [], {}, Counter()
    iterator = jsonl(a.retrieval_cache)
    labels = None
    for sample in samples:
        if not sample.question or tuple(sorted(sample.options or {})) != tuple('ABCD') or sample.answer not in sample.options:
            raise RuntimeError(f'Invalid MCQ: {sample.id}')
        item = next(iterator, None)
        if (item is None or item.get('sample_id') != sample.id or item.get('dataset') != sample.dataset
                or item.get('query_text') != sample.question or item.get('row_idx') != sample.row_idx):
            raise RuntimeError(f'Retrieval/benchmark alignment failure: {sample.id}')
        documents = item.get('documents', [])
        if len(documents) != 32 or any(d.get('rerank_rank') != i+1 or not d.get('text')
                                      or not d.get('corpus_id') for i, d in enumerate(documents)):
            raise RuntimeError(f'Invalid CE Top-32: {sample.id}')
        priors = medcpt_priors(documents)
        if sample.id not in selected:
            continue
        row = {'sample_id': sample.id, 'dataset': sample.dataset, 'question': sample.question,
               'options': sample.options, 'documents': documents[:8], 'priors': priors, 'lengths': {}}
        for mode in MODES:
            prompts = [render_prompt(tokenizer, row, d, mode) for d in [None, *row['documents']]]
            lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False).input_ids]
            reserve = 1 if mode == 'direct' else a.max_rationale_tokens + len(tokenizer.encode('\nFinal answer:', add_special_tokens=False)) + 1
            if max(lengths) + reserve > a.max_model_len:
                raise RuntimeError(f'Context overflow before GPU load: {sample.id}/{mode}; no document truncation allowed')
            row['lengths'][mode] = max(lengths)
            reference = prompts[0] if mode == 'direct' else prompts[0] + '\nFinal answer:'
            prefix = tokenizer.encode(reference, add_special_tokens=False)
            current = []
            for label in 'ABCD':
                full = tokenizer.encode(reference+' '+label, add_special_tokens=False)
                if full[:-1] != prefix or len(full) != len(prefix)+1:
                    raise RuntimeError(f'Unstable single-token label {label}: {sample.id}/{mode}')
                current.append(full[-1])
            if labels is not None and current != labels:
                raise RuntimeError('Choice-label token IDs vary across modes/prompts')
            labels = current
        gold[sample.id] = sample.answer  # Never passed to BatchedPCED.
        counts[sample.dataset] += 1
        rows.append(row)
    if next(iterator, None) is not None:
        raise RuntimeError('Extra rows in retrieval cache')
    return rows, gold, labels, dict(counts)


def make_contract(a, rows, labels):
    source = [Path(__file__), ROOT/'medrag/generation/question_only_pced.py',
              ROOT/'medrag/benchmark.py', ROOT/'medrag/training/graph_input_progress.py',
              ROOT/'medrag/training/graph_input_data.py']
    required = [a.retrieval_cache, a.retrieval_cache.parent/'manifest.json', a.model/'config.json',
                a.model/'tokenizer.json', a.model/'tokenizer_config.json']
    required += [a.benchmark_root/d/'test.jsonl' for d in DATASETS]
    weights = sorted(a.model.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('No local model weights')
    if (a.model/'generation_config.json').exists():
        required.append(a.model/'generation_config.json')
    return {'version': VERSION, 'hypothesis': 'PCED with shared rationale tokens versus one-step label PCED',
            'modes': list(MODES), 'sample_ids': [r['sample_id'] for r in rows], 'split': 'TEST, no tuning',
            'model': str(a.model), 'weights': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in weights},
            'inputs_sha256': {str(p): sha_file(p) for p in required},
            'code_sha256': {str(p.relative_to(ROOT)): sha_file(p) for p in source},
            'git_at_creation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'top_k': 8, 'order': 'question-options-single_evidence; immutable CE rank',
            'retrieval': 'question-only raw IP balanced Top-32, cached CE reranking, first 8; no new retrieval',
            'prior': 'Top-32 raw IP minmax (ties=1), CE sigmoid, harmonic mean; clips epsilon=1e-8; NOT BGE normalization',
            'beta': 'initial mean expert-amateur full-vocabulary Jensen-Shannon divergence, natural log; fixed per mode/question',
            'score': '(1+beta)*expert_logits-beta*amateur_logits+gamma*log(prior); max experts then max tokens',
            'gamma': a.gamma, 'systems': [DIRECT_SYSTEM, RATIONALE_SYSTEM], 'label_ids': labels,
            'generation': {'greedy': True, 'seed': 42, 'dtype': 'bfloat16', 'attention': 'sdpa',
                           'max_model_len': a.max_model_len, 'max_rationale_tokens': a.max_rationale_tokens,
                           'answer': 'one space-prefixed A/B/C/D token, selected by PCED',
                           'terminal': 'natural Final answer marker; EOS/budget append marker with cached KV, then constrained label',
                           'history': 'identical selected token appended to amateur and all experts; independent KV'},
            'runtime': {'memory_fraction': a.memory_fraction, 'max_batch_questions': a.max_batch_questions,
                        'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'),
                        'packages': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'tokenizers')}},
            'overflow': 'fail, never truncate', 'primary': 'paired rationale-minus-direct micro accuracy',
            'secondary': 'dataset accuracy, C-W/W-C counts, bootstrap CI, output-budget hit rate, expert switches, throughput',
            'limitations': 'Output instructions, beta and compute budget necessarily differ; not an isolated causal test of token circulation. No precomputed document KV or paper TTFT claim.'}


def output_path(a, mode, sid):
    return a.output_root / mode / (hashlib.sha256(sid.encode()).hexdigest()+'.json')


def load_done(a, mode, rows, contract_hash):
    done = {}
    for row in rows:
        path = output_path(a, mode, row['sample_id'])
        if path.exists():
            result = json.loads(path.read_text())
            if (result.get('contract_sha256') != contract_hash or result.get('sample_id') != row['sample_id']
                    or result.get('mode') != mode or result.get('answer') not in 'ABCD'
                    or len(result.get('answer', '')) != 1):
                raise RuntimeError(f'Incompatible/corrupt result: {path}')
            done[row['sample_id']] = result
    return done


def summary(rows, gold, results):
    import numpy as np
    conditions = {}
    for mode, records in results.items():
        groups = {'overall': rows, **{d: [r for r in rows if r['dataset'] == d] for d in sorted({r['dataset'] for r in rows})}}
        conditions[mode] = {}
        for name, subset in groups.items():
            n = len(subset)
            c = sum(records[r['sample_id']]['answer'] == gold[r['sample_id']] for r in subset)
            conditions[mode][name] = {'n': n, 'correct': c, 'accuracy': c/n}
        conditions[mode]['finish_reasons'] = dict(Counter(v['finish_reason'] for v in records.values()))
        conditions[mode]['mean_rationale_tokens'] = sum(len(v['rationale_token_ids']) for v in records.values())/len(rows)
        conditions[mode]['mean_winner_switches'] = sum(v['winner_switches'] for v in records.values())/len(rows)
    left = np.array([results['direct'][r['sample_id']]['answer'] == gold[r['sample_id']] for r in rows])
    right = np.array([results['rationale'][r['sample_id']]['answer'] == gold[r['sample_id']] for r in rows])
    diff = right.astype(float)-left.astype(float)
    rng = np.random.default_rng(42)
    boot = [float(diff[rng.integers(len(diff), size=len(diff))].mean()) for _ in range(2000)]
    transitions = {f'{a}->{b}': int(((left == (a == 'C')) & (right == (b == 'C'))).sum())
                   for a in ('C', 'W') for b in ('C', 'W')}
    return {'n': len(rows), 'missing': 0, 'conditions': conditions,
            'paired': {'baseline': 'direct', 'comparison': 'rationale', 'transitions': transitions,
                       'transition_rates_all_questions': {k: v/len(rows) for k, v in transitions.items()},
                       'accuracy_delta': float(diff.mean()), 'bootstrap_95_ci': np.quantile(boot, [.025, .975]).tolist()},
            'interpretation': 'Same model/documents/PCED policy; different output mode and generation budget. Not paper-exact BGE or an isolated recurrence-effect test.'}


def run(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error()
    progress = Progress(['CPU contract/tokenize', 'load frozen Llama', 'direct TEST', 'rationale TEST', 'paired summary'])
    progress.weights[-1] = 1.0  # Small fixed-cost JSON/paired bootstrap stage; recalibrated on completion.
    stage = 'CPU preflight'
    try:
        progress.stage(0, 1)
        manifest = json.loads((a.retrieval_cache.parent/'manifest.json').read_text())
        if (manifest.get('ce', {}).get('question_only') is not True
                or manifest.get('experiment') != 'question_only_ip_fourway_test6545_v1'
                or a.retrieval_cache.name != 'balanced_ce.jsonl'):
            raise RuntimeError('This version requires the frozen balanced CE question-only cache')
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        rows, gold, labels, counts = prepare(a, tokenizer)
        contract = make_contract(a, rows, labels)
        # Git HEAD may advance after this source was committed. Source hashes,
        # rather than HEAD alone, determine compatibility on resume.
        manifest_path = a.output_root/'manifest.json'
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())['contract']
            contract['git_at_creation'] = old['git_at_creation']
        h = ensure_contract(manifest_path, contract)
        atomic_json(a.output_root/'preflight.json', {'n': len(rows), 'datasets': counts, 'label_ids': labels,
                    'max_prompt_tokens': {m: max(r['lengths'][m] for r in rows) for m in MODES}})
        results = {m: load_done(a, m, rows, h) for m in MODES}
        progress.complete(f'N={len(rows)}; {counts}; cached={ {m: len(v) for m,v in results.items()} }; {a.output_root}')
        progress.log('[plan] fixed question-only Top-8 -> frozen Llama -> direct PCED -> shared-token rationale PCED -> paired summary; no learning/retrieval change')
        progress.log('[ETA] Separate mode throughput; direct has 9 prefills/q, rationale additionally has up to 512 shared-token decode steps/q. Overall ETA calibrates after both mode rates exist.')
        if a.preflight_only:
            progress.log('[preflight passed] CPU inputs/prompts/labels/context/contracts checked; no model loaded, no GPU used.')
            return
        stage = 'model load'
        progress.stage(1, 1)
        pending_any = any(len(results[m]) < len(rows) for m in MODES)
        if pending_any:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; use --preflight-only for CPU validation')
            torch.manual_seed(42)
            random.seed(42)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
            model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.bfloat16,
                                                       attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            engine = BatchedPCED(model, tokenizer, labels, gamma=a.gamma, max_rationale_tokens=a.max_rationale_tokens)
        progress.complete('one frozen model shared by all experts/modes; only last-token logits; no LoRA')
        runtime_path = a.output_root/'runtime.json'
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        for index, mode in enumerate(MODES, 2):
            if runtime.get(mode, {}).get('seconds_per_question'):
                progress.weights[index] = runtime[mode]['seconds_per_question'] * len(rows)
        for index, mode in enumerate(MODES, 2):
            stage = mode
            pending = sorted((r for r in rows if r['sample_id'] not in results[mode]), key=lambda r: r['lengths'][mode])
            progress.stage(index, len(rows), initial=len(results[mode]), unit='q', kind='generation')
            offset, cap, fresh, elapsed = 0, a.max_batch_questions, 0, 0.0
            while offset < len(pending):
                window = pending[offset:offset+cap]
                free, total = torch.cuda.mem_get_info()
                allocated = torch.cuda.memory_allocated()
                available = max(0, min(free, total*a.memory_fraction-allocated) - 6*1024**3)
                kv = kv_bytes_per_question(model.config, max(r['lengths'][mode] for r in window), a.max_rationale_tokens+16)
                # Direct mode retains no KV and needs no decode headroom. Both
                # paths still roll back a whole uncommitted batch on OOM.
                batch_n = (len(window) if mode == 'direct' else
                           min(len(window), max(1, int(available*.5/max(1, kv)))))
                batch = window[:batch_n]
                stage = f'{mode} batch first={batch[0]["sample_id"]}; completed={len(results[mode])}/{len(rows)}'
                started = time.monotonic()
                try:
                    generated = engine.generate(batch, mode)
                except torch.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if batch_n == 1:
                        raise
                    cap = max(1, batch_n//2)
                    progress.log(f'[OOM retry] {mode}: uncommitted batch {batch_n} -> {cap}; tokens/documents unchanged')
                    continue
                elapsed += time.monotonic()-started
                fresh += len(batch)
                for value in generated:
                    value['contract_sha256'] = h
                    atomic_json(output_path(a, mode, value['sample_id']), value)
                    results[mode][value['sample_id']] = value
                offset += len(batch)
                progress.update(len(batch))
                runtime[mode] = {'fresh_questions': fresh, 'seconds': elapsed, 'seconds_per_question': elapsed/fresh,
                                 'last_batch_questions': len(batch), 'streams_per_question': 9,
                                 'gpu': torch.cuda.get_device_name(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                atomic_json(runtime_path, runtime)
            progress.complete(f'completed={len(results[mode])}/{len(rows)}; {a.output_root / mode}')
        stage = 'paired summary'
        progress.stage(4, 1)
        report = summary(rows, gold, results)
        atomic_json(a.output_root/'summary.json', report)
        atomic_jsonl(a.output_root/'paired_predictions.jsonl', (
            {'sample_id': r['sample_id'], 'dataset': r['dataset'], 'gold': gold[r['sample_id']],
             **{m: results[m][r['sample_id']]['answer'] for m in MODES}} for r in rows))
        progress.complete(str(a.output_root/'summary.json'))
        progress.log('[complete] '+json.dumps(report['paired'], ensure_ascii=False))
    except Exception as exc:
        progress.log(f'[FAILED] stage={stage}; durable={a.output_root}; {type(exc).__name__}: {exc}. Identical command resumes validated question records; no old results deleted.')
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    run(arguments())

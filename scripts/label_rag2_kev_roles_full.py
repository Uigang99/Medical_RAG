#!/usr/bin/env python3
"""Frozen P3 main + secondary labels on the existing TEST6545 Top-8 union."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.evidence_graph_runtime import LocalDecisions
from medrag.kev_role_labeling import RoleStore, tune_batch
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, identity, sha_file
from medrag.training.graph_input_progress import Progress
from scripts.pilot_rag2_kev_primary_roles import make_job, ROLES

PROMPT_HASH = 'e64491e789710873dd68eae2f4eb6ec314341b4c19b1044ca16dc6b8d234a772'
N = 283462


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', action='store_true', help='CPU-only; no model load or labeling')
    p.add_argument('--candidate-root', type=Path, default=ROOT/'results/rag2_common_pair_candidates_test6545_v1')
    p.add_argument('--output-root', type=Path, default=ROOT/'results/rag2_kev_roles_p3_test6545_v1')
    p.add_argument('--chunk-size', type=int, default=1024)
    p.add_argument('--max-batch-rows', type=int, choices=(48,96,192), default=192)
    p.add_argument('--memory-fraction', type=float, default=.88)
    a = p.parse_args(argv)
    if not 192 <= a.chunk_size <= 4096 or not .5 <= a.memory_fraction <= .92:
        p.error('chunk-size must be 192..4096; memory-fraction .5...92')
    return a


def read_source(a):
    path = a.candidate_root/'candidates.sqlite3'
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT key,value FROM meta')}
    manifest = json.loads((a.candidate_root/'manifest.json').read_text())
    if (not meta.get('inventory_done') or meta['contract_hash'] != manifest['contract_sha256']
            or digest(manifest['contract']) != manifest['contract_sha256']):
        raise ValueError('Source inventory/manifest is incomplete or incompatible')
    stats = meta['inventory_stats']
    count = tuple(db.execute('SELECT count(*),min(idx),max(idx) FROM proposition').fetchone())
    if count != (N,0,N-1) or db.execute('SELECT count(*) FROM question').fetchone()[0] != 6545:
        raise ValueError(f'Expected all TEST6545 propositions, including isolated ones: {count}')
    if stats['unique_propositions'] != N or stats['questions'] != 6545:
        raise ValueError('Source inventory count mismatch')
    # Fast exact input fingerprint, including IDs, verbatim text, and provenance.
    hasher = hashlib.sha256()
    for row in db.execute('SELECT idx,pid,did,claim FROM proposition ORDER BY idx'):
        if not row['claim'].strip():
            raise ValueError(f'Empty proposition: {row["pid"]}')
        hasher.update(json.dumps(tuple(row), ensure_ascii=False).encode()+b'\n')
    return db, stats, {'manifest': manifest['contract_sha256'],
                       'propositions_sha256': hasher.hexdigest(), 'database':str(path.resolve())}


def contract(a, source):
    prompt = ROOT/'docs/kev_primary_roles_p3.json'
    if sha_file(prompt) != PROMPT_HASH:
        raise ValueError('P3 prompt changed: TEST corpus must not be used for prompt tuning')
    models, runtime = ROOT.parent/'models', ROOT.parent/'tools/kev-runtime'
    files = {'prompt': prompt, 'runner': Path(__file__),
             'helper': ROOT/'medrag/kev_role_labeling.py',
             'job_builder': ROOT/'scripts/pilot_rag2_kev_primary_roles.py',
             'role_constants': ROOT/'scripts/pilot_rag2_kev_information_roles.py',
             'runtime': ROOT/'medrag/evidence_graph_runtime.py',
             'kev_model': runtime/'kev/model.py', 'kev_api':runtime/'kev/api.py',
             'kev_checkpoint': runtime/'kev/checkpoint.py',
             'adapter': models/'kev-9b/adapter_model.safetensors',
             'head':models/'kev-9b/head.pt', 'adapter_config':models/'kev-9b/adapter_config.json',
             'base_config':models/'Qwen3.5-9B-Base/config.json'}
    for folder in (models/'kev-9b', models/'Qwen3.5-9B-Base'):
        for pattern in ('*token*json','vocab.json','merges.txt','*.safetensors'):
            for f in sorted(folder.glob(pattern)):
                files[f'{folder.name}/{f.name}'] = f
    if not list((models/'Qwen3.5-9B-Base').glob('*.safetensors')):
        raise FileNotFoundError('Local Qwen base weights missing')
    inputs = {k:identity(v) for k,v in files.items()}
    # Content identities for code/config, not mtimes; no commit-only resume break.
    for value in inputs.values():
        if 'sha256' in value:
            value.pop('mtime_ns', None)
    c = {'version':'kev_roles_full_p3_v1', 'source':source, 'inputs':inputs,
         'scope':'All unique propositions in fixed TEST6545 Top-8; no QA, retrieval change, training, or reference labels',
         'model':'KEV9B + local Qwen3.5-9B-Base; merged frozen adapter',
         'prompt':'P3 frozen before independent confirmation; proposition only; unchanged main then secondary',
         'outputs':'one main role, remaining YES/NO/UNCERTAIN roles, complete probabilities, verbatim source text',
         'expected_questions':6545, 'expected_propositions':N, 'seed':0,
         'generation':'none; argmax fixed-choice probabilities; no new confidence thresholds',
         'dtype':'bfloat16', 'gpu':1, 'shared_prefix':False,
         'batch_policy':{'reference_rows':48,'max_rows':a.max_batch_rows,
                         'memory_fraction':a.memory_fraction,'chunk':a.chunk_size,
                         'selection':'fastest parity-passing calibration, primary/secondary separately'},
         'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','peft')}}
    old = a.output_root/'manifest.json'
    if old.exists() and json.loads(old.read_text()) != {'contract':c,'contract_sha256':digest(c)}:
        raise RuntimeError(f'Existing output has another contract: {old}; select a NEW output-root')
    parent = a.output_root
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 5*1024**3:
        raise RuntimeError('Need at least 5 GiB free for database and atomic export')
    return c, json.loads(prompt.read_text()), runtime, models


def preflight_budget(db, spec, runtime, models):
    """Exact KEV request schema and CPU token budget on the longest 16 texts."""
    sys.path.insert(0, str(runtime))
    from kev.api import SystemOneRequest, to_record
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models/'kev-9b', local_files_only=True)
    largest = 0
    for item in db.execute('SELECT pid,claim FROM proposition ORDER BY length(claim) DESC LIMIT 16'):
        row = {'id':item['pid'],'text':item['claim']}
        for main in (None, *ROLES, 'OTHER', 'UNCERTAIN'):
            job = make_job(row, spec, main)
            rec,_ = to_record(SystemOneRequest.model_validate(job['body']))
            n = len(tokenizer.encode(json.dumps(rec)))
            largest = max(n, largest)
            if n > 7000:
                raise ValueError('Potential KEV truncation on source long text; no input changed')
    return largest


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Use the shell launcher: physical GPU1 only')
    progress = Progress(('CPU contract', 'load KEV', 'TEST role labels', 'export / coverage'))
    db = store = None
    started = time.monotonic()
    active = None
    done = 0
    try:
        progress.log('[plan] TEST6545 Top-8 union -> frozen P3 main+secondary -> atomic labels; no QA evaluation or new training')
        progress.stage(0,1)
        db, stats, source = read_source(a)
        c,spec,runtime,models = contract(a,source)
        budget = preflight_budget(db,spec,runtime,models)
        progress.update()
        progress.complete(f'{N} propositions; original empty document slots={stats["empty_document_slots"]}; longest-input CPU check={budget} tokens')
        if a.preflight:
            progress.log('[preflight passed] No GPU/model loaded; all source counts and P3 schema checked; output untouched')
            return
        a.output_root.mkdir(parents=True, exist_ok=True)
        with (a.output_root/'run.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            h = ensure_contract(a.output_root/'manifest.json',c)
            atomic_json(a.output_root/'prompt.json',spec)
            atomic_json(a.output_root/'invocation.json',{'git_commit':subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'argv':sys.argv,
                'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'contract':h})
            store = RoleStore(a.output_root,h)
            done = store.count()
            if not 0 <= done <= N:
                raise ValueError('Invalid completed label count')
            progress.log(f'[resume] completed={done}/{N}; remaining={N-done}; unique-text decisions cached; original IDs retained')
            progress.log('[ETA] Compatible batch48 pilot implies 17-22h from empty cache; larger-batch gains unmeasured; calibrate first completed chunk')
            progress.weights = [time.monotonic()-started, 8., N/4.6, 30.]
            progress.stage(1,1,initial=int(done==N))
            judge = None
            if done < N:
                import torch
                from transformers.utils import logging
                logging.set_verbosity_error()
                torch.manual_seed(0)
                torch.cuda.set_per_process_memory_fraction(a.memory_fraction,0)
                judge = LocalDecisions(runtime,models/'kev-9b',models/'Qwen3.5-9B-Base',store,batch_rows=48)
                progress.update()
            progress.complete('One resident KEV; bf16; full-row backend; no unsafe hybrid prefix cache')
            progress.stage(2,N,initial=done,unit='prop',kind='kev')
            tuning = {}
            infer_started = time.monotonic()
            inferred = 0
            def score(jobs,phase):
                if phase not in tuning and any(store.get(j) is None for j in jobs):
                    tuning[phase] = tune_batch(judge,jobs,a.max_batch_rows,a.memory_fraction)
                    progress.log(f'[batch] {phase}: {tuning[phase]["selected_rows"]} decision rows; single fallback={judge.batch_disabled}')
                judge.batch_rows = tuning.get(phase,{}).get('selected_rows',48)
                # Parent owns one continuous progress footer; child progress is suppressed.
                return judge.score(jobs,0,lambda *args,**kw:None)
            cursor = db.execute('SELECT idx,pid,did,claim FROM proposition ORDER BY idx')
            while True:
                chunk = cursor.fetchmany(a.chunk_size)
                if not chunk:
                    break
                existing = store.completed(chunk[0]['idx'],chunk[-1]['idx'])
                pending = [x for x in chunk if x['idx'] not in existing]
                if not pending:
                    continue
                active = [pending[0]['idx'],pending[-1]['idx']]
                rows = [{'id':x['pid'],'text':x['claim']} for x in pending]
                main_jobs = [make_job(r,spec) for r in rows]
                main = score(main_jobs,'primary')
                sub_jobs = [make_job(r,spec,main[j['id']]['answers']['primary']['choice'])
                            for r,j in zip(rows,main_jobs)]
                sub = score(sub_jobs,'secondary')
                outputs = []
                for r,j,k in zip(pending,main_jobs,sub_jobs):
                    primary, secondary = main[j['id']]['answers']['primary'], sub[k['id']]['answers']
                    outputs.append({'index':r['idx'],'proposition_id':r['pid'],'document_id':r['did'],
                        'text':r['claim'],'primary':primary,'secondary':secondary,
                        'secondary_labels':[role for role,v in secondary.items() if v['choice']=='YES'],
                        'uncertain_secondary':[role for role,v in secondary.items() if v['choice']=='UNCERTAIN'],
                        'primary_request':j['id'],'secondary_request':k['id'],'contract':h})
                store.save_labels(outputs)
                done += len(outputs)
                inferred += len(outputs)
                progress.update(len(outputs))
                elapsed = time.monotonic()-infer_started
                atomic_json(a.output_root/'RUNTIME.json',{'completed':done,'total':N,'remaining':N-done,
                    'fresh_propositions':inferred,'fresh_wall_seconds':elapsed,
                    'seconds_per_proposition':elapsed/inferred, 'eta_seconds':(N-done)*elapsed/inferred,
                    'calibration':tuning,'oom_splits':judge.oom_splits,'batch_disabled':judge.batch_disabled,
                    'last_committed_index_range':active})
            progress.complete(f'All {done} labeled; durable={a.output_root/"labels.sqlite3"}')
            progress.stage(3,N,unit='prop')
            if store.count() != N:
                raise RuntimeError('Incomplete labels; cannot publish a complete export')
            counts, secondary_counts = Counter(), Counter()
            def export():
                expected = db.execute('SELECT idx,pid,did,claim FROM proposition ORDER BY idx')
                for label, original in zip(store.rows(),expected,strict=True):
                    if (label['index'],label['proposition_id'],label['document_id'],label['text']) != tuple(original):
                        raise ValueError('Export identity/text mismatch')
                    counts[label['primary']['choice']] += 1
                    secondary_counts.update(label['secondary_labels'])
                    progress.update()
                    yield label
            atomic_jsonl(a.output_root/'labels.jsonl',export())
            summary = {'status':'complete','propositions':N,'questions':6545,'missing_labels':0,
                'source_inventory':stats,'primary_distribution':dict(counts),
                'secondary_positive_distribution':dict(secondary_counts),'contract':h,
                'wall_seconds_this_invocation':time.monotonic()-started,
                'quality':'No new human gold or TEST QA evaluation; frozen P3 metadata, not verified semantic relations'}
            atomic_json(a.output_root/'summary.json',summary)
            progress.complete(str(a.output_root/'labels.jsonl'))
            progress.log(f'[complete] {N}/{N}, missing=0; {a.output_root/"summary.json"}')
    except BaseException as exc:
        progress.log(f'[FAILED] stage={progress.index+1}/4 index_range={active}; completed={done}/{N}; remaining={N-done}; durable={a.output_root}; {type(exc).__name__}: {exc}; identical command resumes committed decisions/labels')
        raise
    finally:
        if store is not None:
            store.close()
        if db is not None:
            db.close()
        progress.close()


if __name__ == '__main__':
    run(arguments())

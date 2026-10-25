#!/usr/bin/env python3
"""One-run opt-in: two independent KEV workers on GPU0/1, then checked merge."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import heapq
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import label_rag2_kev_roles_full as worker
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, jsonl
from medrag.training.graph_input_progress import Progress, duration


class DualProgress(Progress):
    remaining = None
    def lines(self):
        first,second = super().lines()
        if self.index == 2:
            overall = None if self.remaining is None else self.remaining + 30
            first = first.rsplit(', ETA ',1)[0] + f', ETA {duration(overall)}]'
            second = second.rsplit(', ETA ',1)[0] + f', ETA {duration(self.remaining)}]'
        return first,second


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--candidate-root',type=Path,default=ROOT/'results/rag2_common_pair_candidates_test6545_v1')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/rag2_kev_roles_p3_test6545_dual_v1')
    p.add_argument('--chunk-size',type=int,default=1024)
    p.add_argument('--max-batch-rows',type=int,choices=(48,96,192),default=192)
    p.add_argument('--memory-fraction',type=float,default=.88)
    a = p.parse_args(argv)
    if not 192 <= a.chunk_size <= 4096 or not .5 <= a.memory_fraction <= .92:
        p.error('Invalid chunk size or memory fraction')
    return a


def worker_args(a,gpu):
    return argparse.Namespace(**{**vars(a),'gpu':gpu,'shard_count':2,'shard_index':gpu,
                                 'output_root':a.output_root/f'gpu{gpu}'})


def merge(a,db,contracts,run_hash,progress):
    """Streaming merge must exactly reconstruct all original IDs/text in order."""
    iterators = []
    for gpu,c in enumerate(contracts):
        root = a.output_root/f'gpu{gpu}'
        summary = json.loads((root/'summary.json').read_text())
        expected = worker.shard(worker_args(a,gpu))[2]
        if (summary.get('status')!='complete' or summary.get('propositions')!=expected
                or summary.get('missing_labels')!=0 or summary.get('contract')!=digest(c)):
            raise ValueError(f'GPU{gpu} incomplete or incompatible: cannot merge')
        iterators.append(jsonl(root/'labels.jsonl'))
    counts = worker.Counter()
    sub_counts = worker.Counter()
    def checked():
        source = db.execute('SELECT idx,pid,did,claim FROM proposition ORDER BY idx')
        merged = heapq.merge(*iterators,key=lambda x:x['index'])
        for label,original in zip(merged,source,strict=True):
            if (label['index'],label['proposition_id'],label['document_id'],label['text']) != tuple(original):
                raise ValueError('Merged duplicate, missing, or altered proposition')
            gpu = label['index'] % 2
            if label['contract'] != digest(contracts[gpu]):
                raise ValueError('Wrong shard contract in merged label')
            counts[label['primary']['choice']] += 1
            sub_counts.update(label['secondary_labels'])
            progress.update()
            yield {**label,'shard_id':gpu,'run_contract':run_hash}
    atomic_jsonl(a.output_root/'labels.jsonl',checked())
    if sum(counts.values()) != worker.N:
        raise ValueError('Merged coverage count mismatch')
    atomic_json(a.output_root/'summary.json',{'status':'complete','propositions':worker.N,
        'questions':6545,'missing_labels':0,'duplicate_ids':0,'contract':run_hash,
        'primary_distribution':dict(counts),'secondary_positive_distribution':dict(sub_counts),
        'shard_counts':[worker.shard(worker_args(a,g))[2] for g in (0,1)],
        'quality':'Frozen P3; no new prompt tuning, gold labels, model training, or QA evaluation'})


def run(a):
    p = DualProgress(('CPU preflight', 'load KEV GPU0+1', 'GPU0+1 labels', 'merge / coverage'))
    children, db = [], None
    try:
        p.log('[plan] Frozen P3 -> even/odd proposition indices -> GPU0 + GPU1 independently -> checked union; no full job starts under --preflight')
        p.stage(0,1)
        db,stats,source = worker.read_source(a)
        contracts = [worker.contract(worker_args(a,g),source)[0] for g in (0,1)]
        _,spec,runtime,models = worker.contract(worker_args(a,0),source)
        worker.preflight_budget(db,spec,runtime,models)
        counts = [worker.shard(worker_args(a,g))[2] for g in (0,1)]
        if sum(counts)!=worker.N:
            raise ValueError('Shard union size mismatch')
        c = {'version':'dual_gpu_p3_v1','worker_contracts':contracts,
             'partition':'idx modulo 2, mutually exclusive and exhaustive',
             'orchestrator_sha256':worker.sha_file(Path(__file__)),
             'purpose':'Execution parallelism only; no classification changes'}
        manifest = a.output_root/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != {'contract':c,'contract_sha256':digest(c)}:
            raise RuntimeError('Dual output contract differs; choose a new output root')
        p.update(); p.complete(f'GPU0={counts[0]}, GPU1={counts[1]}, union={worker.N}; prompts/model/source identical')
        if a.preflight:
            p.log('[preflight passed] CPU only; no output written, no worker/model/GPU launched')
            return
        a.output_root.mkdir(parents=True,exist_ok=True)
        with ExitStack() as stack:
            lock = stack.enter_context((a.output_root/'run.lock').open('a'))
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_hash = ensure_contract(manifest,c)
            p.weights = [8.,60.,worker.N/9.2,30.]
            p.stage(1,2,unit='GPU')
            p.log('[ETA] Approx. 9-12h from empty caches on two idle H200s; not a measured dual-GPU rate. Parent displays maximum remaining worker ETA. Resume each shard independently.')
            for gpu in (0,1):
                wa = worker_args(a,gpu)
                log = stack.enter_context((a.output_root/f'gpu{gpu}.log').open('a',buffering=1))
                env = {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1',
                       'OMP_NUM_THREADS':str(min(8,max(1,(os.cpu_count() or 2)//2)))}
                command = [sys.executable,str(ROOT/'scripts/label_rag2_kev_roles_full.py'),
                    '--candidate-root',str(a.candidate_root),'--output-root',str(wa.output_root),
                    '--gpu',str(gpu),'--shard-count','2','--shard-index',str(gpu),
                    '--chunk-size',str(a.chunk_size),'--max-batch-rows',str(a.max_batch_rows),
                    '--memory-fraction',str(a.memory_fraction)]
                children.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
            labeling = False
            def stop_on_signal(signum,frame):
                raise KeyboardInterrupt(f'Signal {signum}')
            previous = signal.signal(signal.SIGTERM,stop_on_signal)
            stack.callback(signal.signal,signal.SIGTERM,previous)
            while True:
                states = []
                for gpu,child in enumerate(children):
                    code = child.poll()
                    if code not in (None,0):
                        raise RuntimeError(f'GPU{gpu} failed (exit {code}); see {a.output_root/f"gpu{gpu}.log"}')
                    path = a.output_root/f'gpu{gpu}/STATUS.json'
                    states.append(json.loads(path.read_text()) if path.exists() else {})
                ready = sum(s.get('stage') in ('labeling','exporting','complete') for s in states)
                done = sum(s.get('completed',0) for s in states)
                if not labeling:
                    if ready > p.done:
                        p.update(ready-p.done)
                    if ready==2:
                        p.complete('Both KEV workers ready; independent caches/logs; no inter-GPU gradient or communication')
                        p.stage(2,worker.N,initial=done,unit='prop')
                        labeling=True
                if labeling:
                    etas = [None if 'eta_seconds' not in s else max(0,s['eta_seconds']-(time.time()-s['updated_at'])) for s in states]
                    p.remaining = max(etas) if all(x is not None for x in etas) else None
                    if done > p.done:
                        p.update(done-p.done)
                if all(child.poll()==0 for child in children):
                    break
                time.sleep(2)
            p.complete(f'{done}/{worker.N} completed across GPU0/1; worker logs={a.output_root}')
            p.stage(3,worker.N,unit='prop')
            merge(a,db,contracts,run_hash,p)
            p.complete(str(a.output_root/'labels.jsonl'))
            p.log(f'[complete] {worker.N} propositions; missing=0, duplicate IDs=0; {a.output_root/"summary.json"}')
    except BaseException as exc:
        p.log(f'[FAILED] stage={p.index+1}/4; {exc}; durable={a.output_root}; rerun identical command to resume both shards and merge')
        raise
    finally:
        # Only terminate child processes owned by THIS invocation.
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        if db is not None: db.close()
        p.close()


if __name__=='__main__':
    run(arguments())

"""Bounded-memory inference/cache helpers for frozen P3 proposition roles."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

from medrag.training.graph_input_data import digest


def validate_answers(job, answers):
    questions = job['body']['questions']
    if set(answers) != set(questions):
        raise ValueError('Decision keys do not match the frozen request')
    for key, answer in answers.items():
        scores = answer['probabilities']
        if (set(scores) != set(questions[key]['criteria'])
                or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values())
                or abs(sum(scores.values()) - 1) > 1e-4
                or answer['choice'] != max(scores, key=scores.get)
                or abs(answer['confidence'] - scores[answer['choice']]) > 1e-6):
            raise ValueError('Invalid decision distribution or argmax')


class RoleStore:
    """SQLite avoids hundreds of thousands of small, duplicated-prompt files.

    Decisions commit every 32 requests and at every completed proposition chunk.
    Only a chunk with BOTH stages complete is published as labeled propositions.
    An interruption can require recomputing at most 31 uncommitted requests.
    """
    def __init__(self, root, contract_hash):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'labels.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS decision(id TEXT PRIMARY KEY,body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS label(idx INTEGER PRIMARY KEY,pid TEXT UNIQUE NOT NULL,body TEXT NOT NULL);
        ''')
        old = self.db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
        if old and old[0] != contract_hash:
            self.db.close()
            raise RuntimeError('Label database contract mismatch; use another output root')
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('contract',?)", (contract_hash,))
        self.db.commit()
        self.pending = 0

    def get(self, job):
        if job['id'] != digest(job['body']):
            raise ValueError('Request identity mismatch')
        row = self.db.execute('SELECT body FROM decision WHERE id=?', (job['id'],)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        validate_answers(job, value['answers'])
        return value

    def put(self, job, answers, **extra):
        validate_answers(job, answers)
        if job['id'] != digest(job['body']):
            raise ValueError('Request identity mismatch')
        value = {'request_hash': job['id'], 'answers': answers, **extra}
        self.db.execute('INSERT INTO decision VALUES (?,?)',
                        (job['id'], json.dumps(value, ensure_ascii=False)))
        self.pending += 1
        if self.pending >= 32:
            self.flush()
        return value

    def flush(self):
        self.db.commit()
        self.pending = 0

    def count(self):
        return self.db.execute('SELECT count(*) FROM label').fetchone()[0]

    def completed(self, low, high):
        return {x[0] for x in self.db.execute('SELECT idx FROM label WHERE idx BETWEEN ? AND ?', (low, high))}

    def save_labels(self, rows):
        with self.db:
            self.db.executemany('INSERT INTO label VALUES (?,?,?)',
                                [(r['index'], r['proposition_id'], json.dumps(r, ensure_ascii=False)) for r in rows])
        self.pending = 0

    def rows(self):
        for (body,) in self.db.execute('SELECT body FROM label ORDER BY idx'):
            yield json.loads(body)

    def close(self):
        self.flush()
        self.db.close()


def pack(items, rows_limit):
    batch, rows = [], 0
    for item in items:
        n = len(item[2])
        if batch and rows + n > rows_limit:
            yield batch
            batch, rows = [], 0
        batch.append(item)
        rows += n
    if batch:
        yield batch


def output_parity(reference, candidate):
    """Exact winning labels plus the existing .02 probability tolerance."""
    if len(reference) != len(candidate):
        return False
    for ra, rb in zip(reference, candidate):
        if len(ra) != len(rb):
            return False
        for a, b in zip(ra, rb):
            if (len(a) != len(b) or not a
                    or max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__)
                    or any(not math.isfinite(y) or abs(x-y) > .02 + .02*abs(x) for x,y in zip(a,b))):
                return False
    return True


def tune_batch(judge, jobs, max_rows, memory_fraction):
    """Execution-only calibration, never prompt/label/QA tuning.

    Compare 48/96/192 decision-row batches on IDENTICAL requests. Larger batches
    must retain all sampled winners, pass probability parity and avoid OOM.
    One model stays resident; the existing recursively split OOM path is retained.
    """
    import torch
    unique = {j['id']: j for j in jobs}
    items = [(j, *judge.encode(j)) for j in unique.values()]
    items.sort(key=lambda x: len(x[1]['ids']), reverse=True)
    sample, n = [], 0
    for item in items:
        sample.append(item)
        n += len(item[2])
        if n >= max_rows:
            break
    # Warm up kernels before comparing elapsed times. No labels are committed.
    judge._forward(sample[:1])
    reference = None
    records = []
    selected, fastest = 48, float('inf')
    for limit in (48, 96, 192):
        if limit > max_rows:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        old_oom = judge.oom_splits
        # Verify single-vs-batch independently for primary AND secondary phases.
        judge.parity = None
        torch.cuda.synchronize()
        start = time.monotonic()
        output = [result for batch in pack(sample, limit) for result in judge._forward(batch)]
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if reference is None:
            reference = output
        passed = output_parity(reference, output)
        peak = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        valid = passed and judge.oom_splits == old_oom and peak <= total * memory_fraction
        records.append({'batch_rows': limit, 'seconds': elapsed, 'decision_rows': n,
                        'peak_reserved_gib': peak/1024**3, 'parity': passed,
                        'single_parity': judge.parity, 'valid': valid,
                        'oom_splits': judge.oom_splits-old_oom})
        if valid and elapsed < fastest * .95:
            selected, fastest = limit, elapsed
        if judge.batch_disabled:
            break
    return {'selected_rows': selected, 'trials': records,
            'batch_disabled': judge.batch_disabled,
            'note': 'Calibration parity is sampled, not a guarantee for every BF16 near-tie. OOM splitting remains enabled.'}

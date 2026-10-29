#!/usr/bin/env python3
"""Rebuild exact Top-32 pools from compatible per-source search checkpoints.

No vector scan/re-embedding is required when excluding sources: the Top-32 of
the remaining union is contained in its per-source Top-32. Re-score the new
balanced/global candidate union using the unchanged question-only MedCPT CE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_question_only_fourway_retrieval as base
from medrag.retrieval.inner_product_fourway import _atomic_state, derive_candidate_ids, load_state


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-root', type=Path, default=ROOT / 'databases/run_cache/question_only_ip_fourway_test6545_pmc_scope_v1')
    p.add_argument('--output-root', type=Path, default=ROOT / 'databases/run_cache/question_only_ip_fourway_test6545_pmc_scope_no_pmc_v1')
    p.add_argument('--sources', nargs='+', choices=base.SOURCES, default=['pubmed', 'cpg', 'textbooks'])
    p.add_argument('--balanced-quotas', nargs='+', type=int, default=[11, 11, 10])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    a.parent_root, a.output_root = a.parent_root.resolve(), a.output_root.resolve()
    if (len(set(a.sources)) != len(a.sources) or len(a.sources) != len(a.balanced_quotas) or
            min(a.balanced_quotas) < 1 or sum(a.balanced_quotas) != 32):
        p.error('Distinct sources and matching positive quotas summing to 32 are required')
    if (a.parent_root == a.output_root or a.parent_root in a.output_root.parents or
            a.output_root in a.parent_root.parents):
        p.error('Output must not overlap the parent retrieval directory')
    return a


def prepare(a):
    parent_path = a.parent_root / 'manifest.json'
    parent = json.loads(parent_path.read_text())
    body = {k: v for k, v in parent.items() if k not in ('contract_sha256', 'output_root', 'git_commit')}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != parent['contract_sha256']:
        raise RuntimeError('Parent manifest checksum mismatch')
    if parent['split'] != 'test' or parent['datasets'] != list(base.TEST_DATASETS) or not parent.get('corpus_filter'):
        raise RuntimeError('Expected the completed filtered TEST-6545 parent retrieval')
    ce = parent['ce']
    run_args = argparse.Namespace(
        split='test', datasets=base.TEST_DATASETS, output_root=a.output_root,
        benchmark_root=ROOT / 'datasets/benchmark/mcq/unified',
        query_cache_root=ROOT / 'databases/query_embeddings/medcpt_query_encoder_question_only/mcq/unified',
        vector_db_root=ROOT / 'databases/vector_db/RAG_Square',
        corpus_filter=Path(parent['corpus_filter']['path']), ce_model=Path(ce['model_path']),
        vector_chunk_size=parent['vector_chunk_size'], query_batch_size=parent['query_batch_size'],
        ce_batch_size=ce['batch_size'], ce_question_batch_size=ce['question_batch_size'],
        ce_max_length=ce['max_length'], device='cuda:0', dry_run=a.dry_run)
    samples, _, current, _ = base.preflight(run_args)
    for key in ('question_id_text_sha256', 'dataset_contract', 'source_contract', 'retrieval_metric',
                'corpus_filter', 'ce', 'vector_chunk_size', 'query_batch_size'):
        if current[key] != parent[key]:
            raise RuntimeError(f'Incompatible source-ranking reuse: {key}')
    if any(s not in parent['sources'] for s in a.sources):
        raise RuntimeError('Selected source missing in parent')
    by_source = {s['source']: s for s in current['source_contract']}
    pools, checkpoint_contract = {}, {}
    for source in a.sources:
        path = base.source_checkpoint(a.parent_root, source)
        shard_count = len(by_source[source]['shards'])
        cursor, scores, ids = load_state(path, contract=parent['contract_sha256'],
                                         shape=(len(samples), 32), shard_count=shard_count)
        if (cursor != shard_count or not np.isfinite(scores).all() or (ids < 0).any() or
                (ids >= by_source[source]['rows']).any() or (np.diff(scores, axis=1) > 0).any() or
                (np.diff(np.sort(ids, axis=1), axis=1) == 0).any()):
            raise RuntimeError(f'Incomplete or invalid source Top-32: {source}')
        if not run_args.eligibility_masks[source][ids].all():
            raise RuntimeError(f'Ineligible IDs in source ranking: {source}')
        pools[source] = scores, ids
        checkpoint_contract[source] = {'path': str(path), 'sha256': base.sha256_file(path),
                                       'completed_shards': cursor}
    quotas = dict(zip(a.sources, a.balanced_quotas, strict=True))
    for row in range(len(samples)):
        balanced, global_hits = derive_candidate_ids(pools, row, balanced_quotas=quotas)
        if len(balanced) != 32 or len(global_hits) != 32:
            raise RuntimeError(f'Incomplete rebuilt candidate pool: {row}')
    current.pop('contract_sha256')
    current.update(variant='rebuild_selected_sources', sources=a.sources,
                   source_contract=[by_source[s] for s in a.sources],
                   balanced=f'source quotas {quotas}; then descending inner-product merge of 32',
                   balanced_quotas=quotas,
                   hypothesis='Compare the fixed scoped corpus with PMC excluded; balanced source quotas also change',
                   reuse={'parent_manifest': str(parent_path), 'parent_manifest_sha256': base.sha256_file(parent_path),
                          'source_checkpoints': checkpoint_contract,
                          'method': 'exact Top-32 union merge, not post-filtering the old global Top-32'})
    filt = current['corpus_filter']
    filt['by_source'] = {s: filt['by_source'][s] for s in a.sources}
    filt['eligible_rows'] = sum(filt['by_source'].values())
    filt['excluded_sources'] = [s for s in parent['sources'] if s not in a.sources]
    current['code_sha256'][str(Path(__file__).relative_to(ROOT))] = base.sha256_file(Path(__file__))
    hash_body = {k: v for k, v in current.items() if k not in ('output_root', 'git_commit')}
    h = hashlib.sha256(json.dumps(hash_body, sort_keys=True).encode()).hexdigest()
    current['contract_sha256'] = h
    if (a.output_root / 'manifest.json').exists():
        saved = json.loads((a.output_root / 'manifest.json').read_text())
        if saved['contract_sha256'] != h:
            raise RuntimeError('Output contract changed; choose a new output root')
    return run_args, samples, pools, current, h, quotas


def run(a):
    progress = base.Progress(['validate source reuse', 'rebuild candidate pools', 'load CE / rerank', 'merge four outputs'])
    progress.weights[:] = [5., 1., 423., 14.]
    indexes = {}
    print('[plan] verify filtered TEST6545 -> reuse exact per-source Top-32 -> rebuild global/balanced 32 '
          '-> same MedCPT CE -> four files; no new embeddings, no vector scan, no answer generation', flush=True)
    print('[ETA] ~6-8 min on H200 GPU1; previous compatible CE stage 423s; remaining CE batches recalibrate', flush=True)
    try:
        progress.stage(0, 1, unit='check')
        cfg, samples, pools, contract, h, quotas = prepare(a)
        progress.complete(f"{len(samples)} questions; eligible={contract['corpus_filter']['eligible_rows']:,}; quotas={quotas}")
        if a.dry_run:
            progress.log('[dry-run OK] All source contracts and all rebuilt pools verified; no model/GPU/output writes')
            return
        if shutil.disk_usage(a.output_root.parent).free < 4_000_000_000:
            raise RuntimeError('Need 4 GB free disk for candidate parts and outputs')
        base.checked_manifest(a.output_root, contract)
        progress.stage(1, len(a.sources), unit='source')
        for source in a.sources:
            scores, ids = pools[source]
            cursor = contract['reuse']['source_checkpoints'][source]['completed_shards']
            _atomic_state(base.source_checkpoint(a.output_root, source), contract=h,
                          next_shard=cursor, scores=scores, ids=ids)
            indexes[source] = base.FaissSourceIndex(cfg.vector_db_root / source,
                cache_index=False, mmap_index=True, metadata_row_cache_size=50000)
            progress.update(1)
        progress.complete(f'exact cached source rankings; no FAISS vector files loaded; {a.output_root}')
        base.rerank_parts(cfg, samples, pools, indexes, h, progress, stage_index=2, balanced_quotas=quotas)
        base.merge_outputs(cfg, samples, h, progress, stage_index=3)
        summary_path = a.output_root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary.update(sources=a.sources, balanced_quotas=quotas,
                       eligible_rows=contract['corpus_filter']['eligible_rows'],
                       source_rows=contract['corpus_filter']['by_source'], parent_root=str(a.parent_root))
        base.atomic_json(summary_path, summary)
        progress.log(f'[complete] 6545 x four Top-32 sets; selected sources={a.sources}; {a.output_root}')
    except Exception as e:
        progress.log(f'[FAILED] stage={progress.index+1}/4 completed={progress.done}/{progress.total}; '
                     f'remaining={progress.total-progress.done}; durable={a.output_root}; '
                     f'identical command resumes CE parts; {type(e).__name__}: {e}')
        raise
    finally:
        for index in indexes.values():
            index.close()
        progress.close()


if __name__ == '__main__':
    run(arguments())

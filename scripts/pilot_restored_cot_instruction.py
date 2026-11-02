#!/usr/bin/env python3
"""Fixed output-instruction ablation; reuse frozen decoders and paired caches."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import evaluate_question_focused_cot as reference
from medrag.generation import restored_cot_instruction as restored
from medrag.generation.question_only_pced import kv_bytes_per_question
from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest, ensure_contract, jsonl, sha_file
from medrag.training.graph_input_progress import Progress

VERSION = 'restored_cot_instruction_v1'
ARMS = reference.ARMS


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=256, help='Fixed SHA256(42:sample_id) subset; 0 means all TEST')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--max-batch-questions', type=int, default=64)
    p.add_argument('--pced-batch-questions', type=int, default=32)
    p.add_argument('--memory-fraction', type=float, default=.88)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(argv)
    if not 0 <= a.limit <= 6545 or min(a.max_batch_questions, a.pced_batch_questions) < 1 or not 0 < a.memory_fraction < .98:
        p.error('Invalid cohort/runtime arguments')
    a.pced_root = ROOT/'results/question_only_pced_medcpt_two_modes_v1'
    a.previous_root = ROOT/'results/question_only_cot_baselines_pced_matched_v1'
    a.focused_root = ROOT/'results/question_focused_cot_test6545_v1'
    a.output_root = (a.output_root or ROOT/'results'/f'{VERSION}_test{a.limit or 6545}').resolve()
    for old in (a.pced_root, a.previous_root, a.focused_root):
        if old == a.output_root or old in a.output_root.parents or a.output_root in old.parents:
            p.error('Output must be separate from immutable reference results')
    return a


def load_focused(a, rows, pced_hash):
    m = json.loads((a.focused_root/'manifest.json').read_text()); c = m['contract']
    if digest(c) != m['contract_sha256'] or c['version'] != reference.VERSION or c['reference_pced_sha256'] != pced_hash:
        raise RuntimeError('Focused reference contract mismatch')
    for path, h in c['code_sha256'].items():
        if sha_file(ROOT/path) != h:
            raise RuntimeError(f'Immutable focused source changed: {path}')
    expected = set(c['sample_ids']); selected = {r['sample_id']: r for r in rows}; result = {}
    for arm in ARMS:
        seen = set(); out = {}
        for path in sorted((a.focused_root/arm/'shards').glob('*.jsonl')):
            for v in jsonl(path):
                sid = v['sample_id']
                if sid in seen or sid not in expected or v['contract_sha256'] != m['contract_sha256']:
                    raise RuntimeError(f'Invalid focused shard: {path}/{sid}')
                seen.add(sid)
                if sid in selected:
                    reference.validate_record(selected[sid], v, arm, m['contract_sha256']); out[sid] = v
        if seen != expected or set(out) != set(selected):
            raise RuntimeError('Focused reference incomplete')
        result[arm] = out
    return m, result


def run(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging
    logging.set_verbosity_error()
    a.output_root.mkdir(parents=True, exist_ok=True)
    with (a.output_root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = Progress(['CPU contracts/prompts', 'load frozen Llama', 'No-RAG restored', 'Base-RAG restored', 'PCED restored', 'paired diagnosis'])
        n = a.limit or 6545
        p.weights = [30., 20., 436.3*n/6545, 1917.2*n/6545, 3697.9*n/6545, 5.]
        stage = 'CPU contracts/prompts'; completed = {}
        try:
            p.log(f'[plan] fixed TEST diagnostic n={n}; cached original/focused -> output instruction restored -> three CoT arms -> paired behavior/Macro-3; GPU1')
            p.log('[scope] Only replace the output-instruction block. No forced rationale/minimum length; decoder and PCED formula unchanged. No automatic expansion or prompt search.')
            p.log('[ETA] For 256 questions, provisional 5-10m on H200 using previous compatible full-CoT rates plus setup; recalibrate per arm. References are cached, not rerun.')
            p.stage(0, 1)
            ref, cfg = reference.matched.verify_reference(a)
            tok = AutoTokenizer.from_pretrained(cfg.model, local_files_only=True, padding_side='left')
            if tok.pad_token_id is None: tok.pad_token = tok.eos_token
            rows, gold, labels, counts, old_pced = reference.matched.prepare(a, cfg, ref, tok)
            original_manifest, original = reference.load_previous(a, rows, old_pced)
            focused_manifest, focused = load_focused(a, rows, ref['contract_sha256'])
            if focused_manifest['contract']['reference_cot_sha256'] != original_manifest['contract_sha256']:
                raise RuntimeError('Original and focused baselines differ')
            final_ids = tok.encode('Final', add_special_tokens=False)
            if len(final_ids) != 1: raise RuntimeError('Expected single first-token Final')
            final_id = final_ids[0]
            reserve = cfg.max_rationale_tokens + len(tok.encode('\nFinal answer:', add_special_tokens=False)) + 1
            for row in rows:
                prompts = {arm: [restored.baseline_prompt(tok, row, arm)] for arm in ARMS[:2]}
                prompts['pced_cot'] = [restored.pced_prompt(tok, row, d) for d in [None, *row['documents']]]
                row['restored_lengths'] = {arm: max(len(tok.encode(s, add_special_tokens=False)) for s in texts) for arm, texts in prompts.items()}
                if max(row['restored_lengths'].values())+reserve > cfg.max_model_len:
                    raise RuntimeError(f'Context overflow: {row["sample_id"]}; no truncation')
            paths = [Path(__file__), Path(restored.__file__), Path(reference.__file__)]
            c = {'version': VERSION, 'sample_ids': [r['sample_id'] for r in rows], 'seed': 42,
                'selection': 'SHA256(42:sample_id); no gold/previous-output filtering',
                'scope': 'TEST-informed diagnostic, not independent final validation',
                'hypothesis': 'Restoring the original output instruction reduces immediate Final and restores nonempty explanations',
                'changed': 'Replace Give a brief explanation ... through Finish ... with the original combined output sentence; includes removing the rather-than-summary clause',
                'fixed': 'Task/evidence policy, body, retrieval, model, backend, budget, decoder, labels, priors, PCED formula; no minimum rationale length',
                'systems': {'no_rag_and_amateur': restored.NO_RAG_SYSTEM, 'rag_and_experts': restored.RAG_SYSTEM},
                'reference_sha256': {'pced': ref['contract_sha256'], 'original_cot': original_manifest['contract_sha256'], 'focused': focused_manifest['contract_sha256']},
                'reference_predictions_sha256': digest({'original': original, 'focused': focused}),
                'model': str(cfg.model), 'generation': focused_manifest['contract']['generation'], 'pced': focused_manifest['contract']['pced'],
                'label_ids': labels, 'final_token_id': final_id,
                'primary_diagnostic': 'Paired first-token Final rate change vs focused; no-explanation rate also reported',
                'accuracy_metric': 'Macro-3, pooled medical MMLU; secondary for this format diagnosis',
                'decision_rule': 'Large recovery only if Base-RAG AND PCED first-Final drop >=20 percentage points, paired CI upper<0, and no-explanation rate decreases; never auto-expand',
                'bootstrap': {'resamples': 5000, 'seed': 42},
                'runtime': {'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'baseline_batch': a.max_batch_questions, 'pced_batch': a.pced_batch_questions, 'memory_fraction': a.memory_fraction, 'packages': ref['contract']['runtime']['packages']},
                'code_sha256': {str(path.relative_to(ROOT)): sha_file(path) for path in paths},
                'git_at_creation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
            manifest = a.output_root/'manifest.json'
            if manifest.exists(): c['git_at_creation'] = json.loads(manifest.read_text())['contract']['git_at_creation']
            h = ensure_contract(manifest, c)
            results = {arm: reference.load_done(a, rows, arm, h) for arm in ARMS}
            completed = {arm: len(v) for arm, v in results.items()}
            atomic_json(a.output_root/'preflight.json', {'n': len(rows), 'datasets': counts, 'cached': completed,
                'final_token_id': final_id, 'max_prompt_tokens': {arm: max(r['restored_lengths'][arm] for r in rows) for arm in ARMS},
                'focused_behavior': {arm: restored.behavior(focused[arm], final_id) for arm in ARMS}})
            p.complete(f'n={len(rows)}; cached={completed}; {a.output_root}')
            if a.preflight_only:
                p.log('[preflight passed] no model loading or GPU generation'); return
            stage = 'load frozen Llama'; p.stage(1, 1)
            if any(len(v) < len(rows) for v in results.values()):
                if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
                torch.manual_seed(42); torch.backends.cuda.enable_cudnn_sdp(False)
                torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
                model = AutoModelForCausalLM.from_pretrained(cfg.model, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
                model.generation_config.pad_token_id = tok.pad_token_id
            p.complete('shared frozen model; independent KV per question/stream')
            rp = a.output_root/'runtime.json'; runtime = json.loads(rp.read_text()) if rp.exists() else {}
            for index, arm in enumerate(ARMS, 2):
                stage = arm; p.stage(index, len(rows), initial=len(results[arm]), unit='q', kind='generation')
                pending = sorted((r for r in rows if r['sample_id'] not in results[arm]), key=lambda r: (r['restored_lengths'][arm], r['sample_id']))
                pc = arm == 'pced_cot'; streams = 9 if pc else 1
                cap = a.pced_batch_questions if pc else a.max_batch_questions
                if pending:
                    engine = restored.RestoredPCED(model, tok, labels, gamma=ref['contract']['gamma'], max_rationale_tokens=512) if pc else restored.RestoredCoT(model, tok, labels, arm=arm, max_rationale_tokens=512)
                    torch.cuda.reset_peak_memory_stats()
                offset = fresh = 0; elapsed = 0.
                while offset < len(pending):
                    window = pending[offset:offset+cap]; free, total = torch.cuda.mem_get_info()
                    available = max(0, min(free, total*a.memory_fraction-torch.cuda.memory_allocated())-6*1024**3)
                    kv = kv_bytes_per_question(model.config, max(r['restored_lengths'][arm] for r in window), 528, streams=streams)
                    count = min(len(window), max(1, int(available*.5/max(1, kv)))); batch = window[:count]
                    stage = f'{arm} first={batch[0]["sample_id"]} remaining={len(pending)-offset}'; start = time.monotonic()
                    try: generated = engine.generate(batch) if pc else engine.generate_plain(batch, arm)
                    except torch.OutOfMemoryError:
                        gc.collect(); torch.cuda.empty_cache()
                        if count == 1: raise
                        cap = max(1, count//2); p.log(f'[OOM retry] uncommitted batch {count}->{cap}'); continue
                    expected = {r['sample_id']: r for r in batch}
                    if len(generated) != count or {v['sample_id'] for v in generated} != set(expected): raise RuntimeError('Incomplete batch')
                    for v in generated:
                        v['arm'] = arm; v['contract_sha256'] = h
                        reference.validate_record(expected[v['sample_id']], v, arm, h)
                    path = a.output_root/arm/'shards'/('batch_'+hashlib.sha256(batch[0]['sample_id'].encode()).hexdigest()+'.jsonl')
                    if path.exists(): raise RuntimeError(f'Refusing overwrite: {path}')
                    atomic_jsonl(path, generated)
                    results[arm].update({v['sample_id']: v for v in generated})
                    offset += count; fresh += count; elapsed += time.monotonic()-start
                    completed[arm] = len(results[arm]); p.update(count)
                    runtime[arm] = {'fresh_questions': fresh, 'seconds': elapsed, 'seconds_per_question': elapsed/fresh,
                        'last_batch_questions': count, 'gpu': torch.cuda.get_device_name(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                    atomic_json(rp, runtime)
                p.complete(f'{len(results[arm])}/{len(rows)}; {a.output_root/arm}')
            stage = 'paired diagnosis'; p.stage(5, 1)
            report = reference.matched.summarize(rows, gold, results); report['version'] = VERSION
            report['original_conditions'] = reference.matched.summarize(rows, gold, original)['conditions']
            report['focused_conditions'] = reference.matched.summarize(rows, gold, focused)['conditions']
            report['accuracy_changes_vs_focused'] = reference.prompt_changes(rows, gold, focused, results)
            report['behavior'] = {name: {arm: restored.behavior(values[arm], final_id) for arm in ARMS} for name, values in [('original', original), ('focused', focused), ('restored', results)]}
            changes = {arm: restored.behavior_change(focused[arm], results[arm], final_id) for arm in ARMS}
            report['behavior_changes_vs_focused'] = changes
            report['large_format_recovery'] = all(changes[arm]['first_final_rate_delta'] <= -.20 and changes[arm]['first_final_rate_delta_95_ci'][1] < 0 and changes[arm]['after']['no_explanation_rate'] < changes[arm]['before']['no_explanation_rate'] for arm in ARMS[1:])
            report['scope'] = c['scope']; report['interpretation'] = 'Output-block ablation, not an isolated test of then or paragraph breaks. Nonempty explanation is not proof of reasoning quality.'
            atomic_json(a.output_root/'summary.json', report)
            atomic_jsonl(a.output_root/'paired_predictions.jsonl', ({'sample_id': r['sample_id'], 'dataset': r['dataset'], 'gold': gold[r['sample_id']],
                'focused': {arm: focused[arm][r['sample_id']]['answer'] for arm in ARMS}, 'restored': {arm: results[arm][r['sample_id']]['answer'] for arm in ARMS}} for r in rows))
            p.complete(str(a.output_root/'summary.json'))
            p.log('[complete] '+json.dumps({'large_format_recovery': report['large_format_recovery'], 'first_final_rates': {arm: changes[arm]['after']['first_final_rate'] for arm in ARMS}}))
        except Exception as exc:
            p.log(f'[FAILED] stage={stage}; completed={completed}; durable={a.output_root}; {type(exc).__name__}: {exc}; identical command resumes committed shards'); raise
        finally: p.close()


if __name__ == '__main__': run(arguments())

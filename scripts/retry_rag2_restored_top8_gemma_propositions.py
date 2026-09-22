#!/usr/bin/env python3
"""Retry only flagged Top-8 proposition documents, preserving the complete original output."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
os.environ.setdefault('VLLM_LOGGING_LEVEL','ERROR')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

from compare_rag2_proposition_models import (
    ROOT, file_sha, sha, rows, save, save_rows, render, clean_completion,
    ready_weights, parse_model_output, TwoLineProgress,
)
from extract_rag2_restored_gemma_propositions import (
    heartbeat, runtime_batch, reduce_batch_after_oom, restore_batch,
    pending_groups, SHARD_SIZE,
)

SOURCE=ROOT/'datasets/kg/rag2_test6545_restored_top8_gemma_faithful_v3_ctx16k_batch32_v1'
OUTPUT=ROOT/'datasets/kg/rag2_test6545_restored_top8_gemma_faithful_v3_repair_v1'
CONFIG=ROOT/'prompts/proposition_standalone_faithful_v3.config.json'
REPAIR_NOTE=(
    'Retry check: Return one complete JSON object without Markdown. Copy each evidence '
    'value character-for-character from one contiguous span of SOURCE PASSAGE; do not '
    'join fragments or insert ellipses. Cover each recoverable factual statement. '
    'If the passage contains only a title, questions, or a table of contents with '
    'no recoverable factual statement, return {"propositions":[]}.'
)
SCHEMA={
    'type':'object',
    'properties':{'propositions':{'type':'array','items':{
        'type':'object',
        'properties':{'claim':{'type':'string'},'evidence':{'type':'string'}},
        'required':['claim','evidence'],'additionalProperties':False,
    }}},
    'required':['propositions'],'additionalProperties':False,
}


def reasons_for_retry(record):
    reasons=[]
    if record['parse_status']=='parse_failure':reasons.append('parse_failure')
    if record['parse_status']=='empty':reasons.append('empty')
    if record['finish_reason']=='length':reasons.append('truncated')
    if any(not p['evidence_exact_match'] for p in record['propositions']):reasons.append('nonexact_evidence')
    if any(e.endswith(('_duplicate','_empty_evidence')) for e in record['parse_errors']):
        reasons.append('invalid_proposition')
    # A damaged source cannot be fixed by generating from that same source.
    return reasons


def load_candidates(path):
    candidates=[];ids=set();stats=Counter()
    for record in rows(path):
        reasons=reasons_for_retry(record)
        if not reasons:continue
        sid=record['stable_id']
        if sid in ids:raise RuntimeError('Duplicate source ID in original Top-8 outputs')
        ids.add(sid)
        text=record['source_text']
        if not text.strip() or sha(text)!=record['text_sha256']:
            raise RuntimeError(f'Original source text/hash mismatch: {sid}')
        candidates.append(dict(stable_id=sid,source=record['source'],source_text=text,
            text_sha256=record['text_sha256'],reasons=reasons,
            original_parse_status=record['parse_status'],
            original_propositions=len(record['propositions']),
            original_exact=sum(p['evidence_exact_match'] for p in record['propositions'])))
        for reason in reasons:stats[reason]+=1
    return candidates,stats


def shard_records(path, selected, contract_hash):
    result=list(rows(path))
    if len(result)!=len(selected):raise RuntimeError(f'Incomplete repair shard: {path}')
    for row,source in zip(result,selected):
        if row['contract_sha256']!=contract_hash or row['stable_id']!=source['stable_id'] or row['text_sha256']!=source['text_sha256']:
            raise RuntimeError(f'Wrong repair shard identity or contract: {path}')
    return result


def machine_valid(row):
    if row['parsed'] is None or row['finish_reason']=='length' or row['parse_errors']:
        return False
    props=row['parsed']['propositions']
    if not props or not all(p['evidence_exact_match'] for p in props):return False
    # Never nominate an empty or shorter rewrite of an existing proposition set.
    return len(props)>=row['original_propositions']


def aggregate(root,candidates,spans,contract_hash,progress):
    counts=Counter();reason_counts=Counter()
    files=['repair_attempts.jsonl','machine_valid_candidates.jsonl','unresolved.jsonl']
    parts=[root/(name+'.partial') for name in files]
    with contextlib.ExitStack() as stack:
        out=[stack.enter_context(part.open('w')) for part in parts]
        for start,stop,path in spans:
            for row,source in zip(shard_records(path,candidates[start:stop],contract_hash),candidates[start:stop]):
                checked=dict(row,machine_valid=machine_valid(row),source_text=source['source_text'])
                out[0].write(json.dumps(checked,ensure_ascii=False)+'\n')
                out[1 if checked['machine_valid'] else 2].write(json.dumps(checked,ensure_ascii=False)+'\n')
                counts['documents']+=1
                counts['machine_valid']+=checked['machine_valid']
                counts['still_empty']+=(row['parsed'] is not None and not row['parsed']['propositions'])
                counts['parse_failure']+=(row['parsed'] is None)
                counts['nonexact']+=any(not p['evidence_exact_match'] for p in (row['parsed'] or {}).get('propositions',[]))
                counts['truncated']+=(row['finish_reason']=='length')
                for reason in source['reasons']:reason_counts[reason]+=1
                progress.done+=1
    for part,name in zip(parts,files):part.replace(root/name)
    summary=dict(counts=counts,original_reasons=reason_counts,contract_sha256=contract_hash,
        interpretation='Machine-valid means schema, length, and exact quotations only; source faithfulness and completeness require separate review. Original outputs are unchanged.')
    save(root/'summary.json',summary)
    save(root/'COMPLETE.json',dict(contract_sha256=contract_hash,documents=len(candidates)))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir',type=Path,default=SOURCE)
    ap.add_argument('--output-dir',type=Path,default=OUTPUT)
    ap.add_argument('--batch-size',type=int,choices=[8,16,32],default=32)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--resolve-output-dir',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--reduce-batch-after-oom',action='store_true',help=argparse.SUPPRESS)
    args=ap.parse_args();root=args.output_dir
    if args.resolve_output_dir:print(root);return
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.reduce_batch_after_oom:
        if not reduce_batch_after_oom(root):raise SystemExit(1)
        return
    expected=4275
    stages=['check original/cache','check prompt lengths','load Gemma31B','retry candidates','write comparison files']
    progress=TwoLineProgress(stages)
    progress.plan(['select original Top-8 errors/empty/nonexact, no source change',
        'check every selected source against 16k context',
        'same checkpoint with JSON schema-constrained output',
        '8-document durable shards; CUDA OOM retries 32->16->8 then restores 32',
        'keep original data intact; save every retry and machine-valid subset'])
    progress.estimates=[10.,None,25.,None,30.]
    try:
        progress.start(1,1,unit='contract')
        original=args.source_dir/'document_propositions.jsonl'
        complete=args.source_dir/'COMPLETE.json'
        config=json.loads(CONFIG.read_text());prompt_path=CONFIG.parent/config['system_prompt_file']
        if file_sha(prompt_path)!=config['system_prompt_file_sha256']:raise RuntimeError('Frozen system prompt changed')
        prompt=prompt_path.read_text().strip();model=ROOT.parent/'models'/config['validated_model']
        if not ready_weights(model):raise RuntimeError('Missing Gemma checkpoint')
        if not original.is_file() or not complete.is_file():raise RuntimeError('Original Top-8 extraction is incomplete')
        original_complete=json.loads(complete.read_text())
        if original_complete['documents']!=44911:raise RuntimeError('Original cohort is not the 44,911-document Top-8 set')
        if shutil.disk_usage(root).free<3*1024**3:raise RuntimeError('At least 3 GiB free space required')
        candidates,reason_counts=load_candidates(original)
        if len(candidates)!=expected:raise RuntimeError(f'Unexpected repair cohort: {len(candidates)} != {expected}; inspect the original version')
        contract=dict(version='top8_proposition_repair_v1',original_path=str(original.resolve()),
            original_sha256=file_sha(original),original_contract_sha256=original_complete['contract_sha256'],
            selected_count=len(candidates),selected_id_sha256=sha(json.dumps([r['stable_id'] for r in candidates])),
            original_reasons=reason_counts,source_prompt_sha256=file_sha(prompt_path),repair_note=REPAIR_NOTE,
            structured_json_schema=SCHEMA,model_path=str(model),model_config_sha256=file_sha(model/'config.json'),
            max_model_len=16384,max_output_tokens=6144,temperature=0.,effective_batch_size=args.batch_size,
            oom_policy='split failed group to 16/8 then restore 32; 8-document immutable shards',
            parser_sha256=file_sha(ROOT/'scripts/extract_rag2_top8_standalone_propositions.py'),
            parser_worker_sha256=file_sha(ROOT/'scripts/compare_rag2_proposition_models.py'),
            environment={n:importlib.metadata.version(n) for n in ['vllm','transformers','torch']})
        contract_hash=sha(json.dumps(contract,sort_keys=True));mf=root/'manifest.json'
        if mf.exists() and json.loads(mf.read_text())['contract_sha256']!=contract_hash:
            raise RuntimeError('Repair output contract mismatch; use another output-dir')
        if not mf.exists():save(mf,dict(contract=contract,contract_sha256=contract_hash))
        batch=runtime_batch(root,contract_hash,args.batch_size)
        recovery=json.loads((root/'oom_batch_limit.json').read_text()) if (root/'oom_batch_limit.json').exists() else {}
        retry_stop=recovery.get('retry_stop') if batch<args.batch_size else None
        spans=[];missing=[];completed=0
        for start in range(0,len(candidates),SHARD_SIZE):
            stop=min(start+SHARD_SIZE,len(candidates));path=root/'generation_shards'/f'repair_{start:05d}_{stop:05d}.jsonl'
            span=(start,stop,path);spans.append(span)
            if path.exists():shard_records(path,candidates[start:stop],contract_hash);completed+=stop-start
            else:missing.append(span)
        if batch<args.batch_size and (retry_stop is None or not any(s<retry_stop for s,_,_ in missing)):
            restore_batch(root,contract_hash,args.batch_size);batch=args.batch_size;retry_stop=None
        progress.complete(f'selected={len(candidates)} cached={completed} remaining={len(candidates)-completed}; reasons={dict(reason_counts)}; output={root}')
        if (root/'COMPLETE.json').exists():
            completed_marker=json.loads((root/'COMPLETE.json').read_text())
            if completed_marker['contract_sha256']!=contract_hash or missing:
                raise RuntimeError('Repair COMPLETE marker disagrees with shards')
            if all((root/name).is_file() for name in ['repair_attempts.jsonl','machine_valid_candidates.jsonl','unresolved.jsonl','summary.json']):
                progress.close(force_complete=True);print('[complete] Repair is already cached');return
        progress.start(2,len(candidates),unit='document')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        max_input=0
        with heartbeat(progress):
            for start in range(0,len(candidates),256):
                rendered=[render(tokenizer,prompt+'\n\n'+REPAIR_NOTE,record['source_text']) for record in candidates[start:start+256]]
                ids=tokenizer(rendered,add_special_tokens=False)['input_ids']
                for record,tokens in zip(candidates[start:start+256],ids):
                    if len(tokens)+6144>16384:
                        raise RuntimeError(f'Repair prompt exceeds context for {record["stable_id"]}: {len(tokens)}+6144')
                max_input=max(max_input,max(map(len,ids)));progress.done=min(start+256,len(candidates))
        progress.complete(f'max_input={max_input}; input+6144<=16384; no truncation')
        if args.preflight_only:
            progress.close();print('[preflight passed] CPU only; GPU repair not started');return
        if missing and os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Set CUDA_VISIBLE_DEVICES=1')
        progress.start(3,1,unit='model',estimate=25. if missing else 0.)
        if missing:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            save(root/'generation_cursor.json',dict(contract_sha256=contract_hash,phase='load',batch_size=batch))
            with heartbeat(progress):
                llm=LLM(model=str(model),dtype='bfloat16',tensor_parallel_size=1,
                    gpu_memory_utilization=0.8,max_model_len=16384,max_num_seqs=batch,
                    max_num_batched_tokens=4096,seed=20260921,enforce_eager=True,
                    enable_prefix_caching=True,language_model_only=True,disable_log_stats=True)
            sampling=SamplingParams(temperature=0.,max_tokens=6144,skip_special_tokens=False,
                structured_outputs=StructuredOutputsParams(json=SCHEMA))
        progress.complete('Gemma loaded' if missing else 'All repair shards cached; GPU skipped')
        progress.start(4,len(candidates),initial=completed,unit='document',estimate=len(candidates)*0.75)
        (root/'generation_shards').mkdir(exist_ok=True)
        with heartbeat(progress):
            for group in pending_groups(missing,batch,retry_stop):
                selected=[r for start,stop,_ in group for r in candidates[start:stop]]
                prompts=[render(tokenizer,prompt+'\n\n'+REPAIR_NOTE,r['source_text']) for r in selected]
                save(root/'generation_cursor.json',dict(contract_sha256=contract_hash,phase='generate',batch_size=batch,
                    start=group[0][0],stop=group[-1][1]))
                tick=time.monotonic();outputs=llm.generate(prompts,sampling,use_tqdm=False);duration=time.monotonic()-tick
                if len(outputs)!=len(selected):raise RuntimeError('Repair output count mismatch')
                generated=[]
                for source,out in zip(selected,outputs):
                    result=out.outputs[0];parsed,errors=parse_model_output(clean_completion(result.text),source['source_text'])
                    generated.append(dict(stable_id=source['stable_id'],source=source['source'],text_sha256=source['text_sha256'],
                        contract_sha256=contract_hash,original_reasons=source['reasons'],
                        original_parse_status=source['original_parse_status'],original_propositions=source['original_propositions'],
                        original_exact=source['original_exact'],raw_output=result.text,parsed=parsed,parse_errors=errors,
                        finish_reason=result.finish_reason,input_tokens=len(out.prompt_token_ids),
                        output_tokens=len(result.token_ids),output_token_ids=list(result.token_ids),
                        batch_seconds=duration,actual_batch_size=len(selected)))
                offset=0
                for start,stop,path in group:
                    count=stop-start;save_rows(path,generated[offset:offset+count]);offset+=count;progress.done+=count
                fresh=progress.done-completed
                if fresh>=32:progress.estimates[3]=len(candidates)*(time.time()-progress.stage_started)/fresh
        if retry_stop is not None:
            restore_batch(root,contract_hash,args.batch_size)
            progress.log(f'[recovery complete] Saved failed group; returning to batch {args.batch_size}; completed={progress.done}/{len(candidates)}')
            raise SystemExit(75)
        progress.complete(f'All {len(candidates)} retries saved under {root}/generation_shards')
        progress.start(5,len(candidates),unit='document',estimate=30.)
        with heartbeat(progress):summary=aggregate(root,candidates,spans,contract_hash,progress)
        progress.complete(str(root/'summary.json'));progress.close(force_complete=True)
        print('[complete] '+json.dumps(summary['counts']))
    except BaseException as exc:
        if isinstance(exc,SystemExit) and exc.code==75:progress.close();raise
        progress.log(f'[FAILED] stage={progress.index}/5 item={progress.done}/{progress.total}; {type(exc).__name__}: {exc}; durable={root}/generation_shards; rerun the same command')
        progress.close();raise


if __name__=='__main__':main()

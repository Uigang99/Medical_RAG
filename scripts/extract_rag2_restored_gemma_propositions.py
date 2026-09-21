#!/usr/bin/env python3
"""Full restored-union extraction with the frozen, pilot-tested Gemma prompt.

Reuses the tested renderer/parser/engine settings; batch shards are immutable.
No retrieval, quality-based filtering or prompt repair. The shell supervisor
restarts an OOM-failed engine at lower concurrency, preserving 8-document shards.
"""
import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import sys
import threading
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

INPUT_ROOT=ROOT/'datasets/kg/rag2_test6545_top1_2_4_8_16_sentence_restored_v1'
OUTPUT=ROOT/'datasets/kg/rag2_test6545_restored_union88294_gemma_standalone_faithful_v3_ctx16k_v1'
CONFIG=ROOT/'prompts/proposition_standalone_faithful_v3.config.json'
EXPECTED={'textbooks':12991,'cpg':18526,'pubmed':27703,'pmc':29074}
SHARD_SIZE=8


def select_top8(docs, membership):
    ids={r['stable_id'] for r in rows(membership) if r['top_k']==8}
    selected=[d for d in docs if d['stable_id'] in ids]
    if len(ids)!=44911 or len(selected)!=44911:
        raise RuntimeError('Top-8 membership must resolve to exactly 44,911 restored documents')
    return selected


def runtime_batch(root, contract_hash, requested):
    path=root/'oom_batch_limit.json'
    if not path.exists():return requested
    state=json.loads(path.read_text())
    if state['contract_sha256']!=contract_hash:
        raise RuntimeError('OOM resume state belongs to a different contract')
    batch=state['batch_size']
    if batch not in (8,16,32) or batch>requested:
        raise RuntimeError('Invalid saved OOM batch limit')
    return batch


def reduce_batch_after_oom(root):
    manifest=json.loads((root/'manifest.json').read_text())
    ch=manifest['contract_sha256']
    planned=manifest['contract'].get('effective_batch_size',8)
    current=runtime_batch(root,ch,planned)
    cursor=json.loads((root/'generation_cursor.json').read_text())
    if cursor['contract_sha256']!=ch or cursor['phase']!='generate':
        print('[OOM] Failure outside generation; batch splitting is not a safe remedy. Completed shards are preserved.',flush=True)
        return False
    if current<=8:
        print('[OOM] batch=8 also failed; stopping safely. Completed shards are preserved.',flush=True)
        return False
    lowered=current//2
    old=json.loads((root/'oom_batch_limit.json').read_text()) if (root/'oom_batch_limit.json').exists() else {}
    retry_stop=old.get('retry_stop') or cursor['stop']
    state=dict(contract_sha256=ch,batch_size=lowered,retry_stop=retry_stop,
        recovery_start=old.get('recovery_start',cursor['start']),reason='CUDA OOM reported by failed worker',updated_at=time.time())
    save(root/'oom_batch_limit.json',state)
    with (root/'oom_events.jsonl').open('a') as f:f.write(json.dumps(dict(event='reduce',previous_batch=current,**state))+'\n')
    print(f'[OOM recovery] batch {current} -> {lowered}; retry only pending rows before {retry_stop}, then restore {planned}; completed shards preserved; output={root}',flush=True)
    return True


def pending_groups(missing, batch, retry_stop=None):
    if retry_stop is not None:missing=[span for span in missing if span[0]<retry_stop]
    for start in range(0,len(missing),batch//SHARD_SIZE):
        yield missing[start:start+batch//SHARD_SIZE]


def restore_batch(root, contract_hash, requested):
    state=dict(contract_sha256=contract_hash,batch_size=requested,retry_stop=None,updated_at=time.time())
    save(root/'oom_batch_limit.json',state)
    with (root/'oom_events.jsonl').open('a') as f:f.write(json.dumps(dict(event='restore',**state))+'\n')


def load_source(path):
    docs=list(rows(path))
    ids=[d['stable_id'] for d in docs]
    if len(ids)!=88294 or len(set(ids))!=88294 or Counter(d['source'] for d in docs)!=EXPECTED:
        raise RuntimeError('Restored union differs from fixed 88,294-document cohort')
    if any(not isinstance(d.get('restored_text'),str) or not d['restored_text'].strip() for d in docs):
        raise RuntimeError('Empty restored document; no GPU loaded')
    return docs


def shard_records(path, docs, start, stop, contract_hash):
    rr=list(rows(path))
    expected=docs[start:stop]
    if len(rr)!=len(expected):raise RuntimeError(f'Incomplete durable shard: {path}')
    for r,d in zip(rr,expected):
        if (r.get('stable_id')!=d['stable_id'] or r.get('contract_sha256')!=contract_hash
                or r.get('text_sha256')!=sha(d['restored_text'])):
            raise RuntimeError(f'Wrong shard identity/contract: {path}')
    return rr


@contextlib.contextmanager
def heartbeat(progress):
    stop=threading.Event()
    def tick():
        while not stop.wait(2):progress.render()
    thread=threading.Thread(target=tick,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join()


def aggregate(root,docs,spans,contract_hash,progress):
    counts=Counter();by_source={k:Counter() for k in EXPECTED}
    names=['document_propositions.jsonl','propositions.jsonl','review_required.jsonl']
    tmp=[root/(n+'.partial') for n in names]
    with contextlib.ExitStack() as stack:
        handles=[stack.enter_context(p.open('w')) for p in tmp]
        for start,stop,path in spans:
            for r,d in zip(shard_records(path,docs,start,stop,contract_hash),docs[start:stop]):
                pp=(r['parsed'] or {}).get('propositions',[])
                # Empty JSON lists are valid; only structural/length/source warnings go to review.
                warnings=[e for e in r['parse_errors'] if e!='empty_propositions']
                if r['finish_reason']=='length':warnings.append('output_truncated')
                if d['status']=='review':warnings.append('source_boundary_review')
                status='parse_failure' if r['parsed'] is None else 'empty' if not pp else 'parsed'
                doc=dict(r,propositions=pp,parse_status=status,review_reasons=warnings,
                    source_text=d['restored_text'],original_text=d['original_text'],
                    context_chunk_ids=d['context_chunk_ids'],source_status=d['status'],source_quality_flags=d['quality_flags'])
                handles[0].write(json.dumps(doc,ensure_ascii=False)+'\n')
                counts['documents']+=1;counts[status]+=1;counts['propositions']+=len(pp)
                counts['truncated']+=r['finish_reason']=='length'
                by_source[d['source']]['documents']+=1;by_source[d['source']]['propositions']+=len(pp)
                if warnings:handles[2].write(json.dumps(doc,ensure_ascii=False)+'\n');counts['review_documents']+=1
                for j,p in enumerate(pp,1):
                    flat=dict(proposition_id=f"{d['stable_id']}::P{j}",stable_id=d['stable_id'],
                        source=d['source'],text_sha256=r['text_sha256'],model=r['model'],
                        prompt_version=r['prompt_version'],source_status=d['status'],
                        generation_finish_reason=r['finish_reason'],**p)
                    handles[1].write(json.dumps(flat,ensure_ascii=False)+'\n')
                    counts['exact_evidence']+=p['evidence_exact_match']
                progress.done+=1
    for p,n in zip(tmp,names):p.replace(root/n)
    result=dict(counts=counts,by_source=by_source,scope=f'Automatic completeness/format audit, not semantic accuracy; all {len(docs):,} selected sources retained',
                contract_sha256=contract_hash)
    save(root/'summary.json',result)
    save(root/'COMPLETE.json',dict(contract_sha256=contract_hash,documents=len(docs)))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=INPUT_ROOT/'restored_chunks.jsonl')
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--top-k',choices=['union','8'],default='union')
    ap.add_argument('--batch-size',type=int,choices=[8,16,32],default=8)
    ap.add_argument('--resolve-output-dir',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--reduce-batch-after-oom',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--max-model-len',type=int,default=16384,
                    help='Context capacity only; prompt and output budget remain frozen. Use a separate output directory if changed.')
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args()
    default=OUTPUT if a.top_k=='union' and a.batch_size==8 else ROOT/'datasets/kg'/f'rag2_test6545_restored_{"top8" if a.top_k=="8" else "union"}_gemma_faithful_v3_ctx16k_batch{a.batch_size}_v1'
    root=a.output_dir or default
    if a.resolve_output_dir:print(root);return
    root.mkdir(parents=True,exist_ok=True)
    # Different concurrent commands must not race over shards or aggregation.
    import fcntl
    lock=(root/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.reduce_batch_after_oom:
        if not reduce_batch_after_oom(root):raise SystemExit(1)
        return
    expected_count=44911 if a.top_k=='8' else 88294
    seconds_per_doc={8:3.1,16:68.1655/64,32:47.0144/64}[a.batch_size]
    p=TwoLineProgress(['validate source/cache','tokenize input budgets','load Gemma31B','generate propositions','write outputs'])
    p.plan([f'{expected_count:,} fixed restored documents; scope={a.top_k}; source-review cases retained',
            'cached or validate all prompt token budgets; no truncation',
            'Gemma31B on GPU 1, context capacity=16384 (pilot8192); output remains4096; load ~25s',
            f'initial batch={a.batch_size}; 8-document shards; on OOM split failed group 32->16->8, then restore initial batch',
            'stream source-linked outputs and warnings'])
    p.estimates=[10.,None,25.,expected_count*seconds_per_doc,90.]
    try:
        p.start(1,1,unit='contract',estimate=10.)
        with heartbeat(p):
            config=json.loads(CONFIG.read_text());g=dict(config['generation'])
            g['max_model_len']=a.max_model_len
            prompt_path=CONFIG.parent/config['system_prompt_file']
            if file_sha(prompt_path)!=config['system_prompt_file_sha256']:raise RuntimeError('Frozen prompt hash mismatch')
            prompt=prompt_path.read_text().strip();model=ROOT.parent/'models'/config['validated_model']
            if not ready_weights(model):raise RuntimeError('Incomplete local Gemma checkpoint')
            model_config=json.loads((model/'config.json').read_text())
            if not 8192<=a.max_model_len<=model_config['text_config']['max_position_embeddings']:
                raise RuntimeError('Requested context capacity outside checkpoint bounds')
            download=json.loads((model/'download_manifest.json').read_text())
            if download['revision']!=config['checkpoint_revision']:raise RuntimeError('Wrong checkpoint revision')
            if shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('Require at least 10 GiB free for outputs')
            docs=load_source(a.input)
            membership=INPUT_ROOT/'question_chunk_membership.jsonl'
            if a.top_k=='8':docs=select_top8(docs,membership)
            contract=dict(version='restored_union88294_gemma_faithful_v3',input_path=str(a.input.resolve()),
                input_sha256=file_sha(a.input),documents=len(docs),by_source=dict(Counter(d['source'] for d in docs)),
                prompt_config=config,model_path=str(model),model_config_sha256=file_sha(model/'config.json'),
                effective_max_model_len=a.max_model_len,
                chat_template_sha256=file_sha(model/'chat_template.jinja'),
                max_num_batched_tokens=4096,skip_special_tokens=False,parser='unchanged pilot parser + envelope cleanup',
                parser_worker_sha256=file_sha(ROOT/'scripts/compare_rag2_proposition_models.py'),
                parser_base_sha256=file_sha(ROOT/'scripts/extract_rag2_top8_standalone_propositions.py'),
                environment={n:importlib.metadata.version(n) for n in ['vllm','transformers','torch']},
                source_review_policy='retain all 58; preserve status and warnings; no silent exclusion',
                retrieval='unchanged balanced 4corpus*k -> rerank k; union k=1,2,4,8,16',
                uses_question_answer_label=False)
            if a.top_k=='8':
                contract.update(version='restored_top8_gemma_faithful_v3',membership_sha256=file_sha(membership),
                    retrieval='unchanged balanced 4corpus*8 -> rerank 8; exact existing Top-8 membership',
                    source_review_policy='retain all selected sources including warnings; no silent exclusion')
            if a.batch_size!=8:contract.update(effective_batch_size=a.batch_size,oom_policy='restart engine; split failed group down to 8; restore initial concurrency after group; immutable 8-document shards')
            ch=sha(json.dumps(contract,sort_keys=True))
            mf=root/'manifest.json'
            if mf.exists() and json.loads(mf.read_text())['contract_sha256']!=ch:
                raise RuntimeError('Output contract differs; choose a new output-dir before GPU loading')
            if not mf.exists():
                from extract_rag2_top8_standalone_propositions import git_commit
                save(mf,dict(contract=contract,contract_sha256=ch,git_commit=git_commit(),
                    runner_sha256=file_sha(Path(__file__)),hypothesis='Scale frozen extraction configuration without changing documents or prompt'))
            batch=runtime_batch(root,ch,a.batch_size)
            recovery=json.loads((root/'oom_batch_limit.json').read_text()) if (root/'oom_batch_limit.json').exists() else {}
            retry_stop=recovery.get('retry_stop') if batch<a.batch_size else None
            seconds_per_doc={8:3.1,16:68.1655/64,32:47.0144/64}[batch]
            p.estimates[3]=len(docs)*seconds_per_doc
            spans=[];completed=0;missing=[]
            for start in range(0,len(docs),SHARD_SIZE):
                stop=min(start+SHARD_SIZE,len(docs));path=root/'generation_shards'/f'batch_{start:07d}_{stop:07d}.jsonl'
                spans.append((start,stop,path))
                if path.exists():shard_records(path,docs,start,stop,ch);completed+=stop-start
                else:missing.append((start,stop,path))
            if batch<a.batch_size and (retry_stop is None or not any(s<retry_stop for s,_,_ in missing)):
                restore_batch(root,ch,a.batch_size);batch=a.batch_size;retry_stop=None
        p.complete(f'cached={completed:,}; remaining={len(docs)-completed:,}; active_batch={batch}; output={root}')
        if (root/'COMPLETE.json').exists():
            complete=json.loads((root/'COMPLETE.json').read_text())
            if complete['contract_sha256']!=ch or missing:raise RuntimeError('COMPLETE marker disagrees with durable shards')
            if all((root/n).is_file() for n in ['document_propositions.jsonl','propositions.jsonl','review_required.jsonl','summary.json']):
                p.close(force_complete=True);print('[complete] All work already cached; no GPU loaded');return
        p.start(2,len(docs),unit='document',estimate=None)
        budget_path=root/'token_budget.json';budget=None
        if budget_path.exists():
            budget=json.loads(budget_path.read_text())
            if budget['contract_sha256']!=ch:raise RuntimeError('Wrong token budget cache')
        if budget is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
            max_tokens=0
            with heartbeat(p):
                for start in range(0,len(docs),512):
                    rendered=[render(tokenizer,prompt,d['restored_text']) for d in docs[start:start+512]]
                    ids=tokenizer(rendered,add_special_tokens=False)['input_ids']
                    for d,t in zip(docs[start:start+512],ids):
                        if len(t)+g['max_output_tokens']>g['max_model_len']:
                            raise RuntimeError(f'Input+output exceeds context; no truncation allowed: {d["stable_id"]}, input={len(t)}')
                    max_tokens=max(max_tokens,max(map(len,ids)));p.done=min(start+512,len(docs))
            budget=dict(contract_sha256=ch,documents=len(docs),max_input_tokens=max_tokens)
            save(budget_path,budget)
        p.complete(f'max_input_tokens={budget["max_input_tokens"]}; max_output=4096; no input truncation')
        if a.preflight_only:
            p.close();print('[preflight passed] CPU only; GPU generation not started');return
        if missing and os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Set CUDA_VISIBLE_DEVICES=1')
        p.start(3,1,unit='model',estimate=25. if missing else 0.)
        if missing:
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
            save(root/'generation_cursor.json',dict(contract_sha256=ch,phase='load',batch_size=batch))
            with heartbeat(p):
                llm=LLM(model=str(model),dtype=g['dtype'],tensor_parallel_size=1,
                    gpu_memory_utilization=g['gpu_memory_utilization'],max_model_len=g['max_model_len'],
                    max_num_seqs=batch,max_num_batched_tokens=4096,seed=g['seed'],
                    enforce_eager=True,enable_prefix_caching=True,language_model_only=True,disable_log_stats=True)
            sampling=SamplingParams(temperature=0.,max_tokens=g['max_output_tokens'],skip_special_tokens=False)
        p.complete('Gemma31B loaded' if missing else 'All shards cached; GPU loading skipped')
        p.start(4,len(docs),initial=completed,unit='document',estimate=len(docs)*seconds_per_doc)
        (root/'generation_shards').mkdir(exist_ok=True)
        with heartbeat(p):
            for group in pending_groups(missing,batch,retry_stop):
                selected=[d for start,stop,_ in group for d in docs[start:stop]]
                prompts=[render(tokenizer,prompt,d['restored_text']) for d in selected]
                save(root/'generation_cursor.json',dict(contract_sha256=ch,phase='generate',batch_size=batch,start=group[0][0],stop=group[-1][1]))
                tick=time.monotonic();generated=llm.generate(prompts,sampling,use_tqdm=False)
                duration=time.monotonic()-tick;rr=[]
                if len(generated)!=len(selected):raise RuntimeError('vLLM output count mismatch')
                for d,out in zip(selected,generated):
                    result=out.outputs[0];parsed,errors=parse_model_output(clean_completion(result.text),d['restored_text'])
                    rr.append(dict(stable_id=d['stable_id'],source=d['source'],doc_id=d.get('doc_id'),
                        text_sha256=sha(d['restored_text']),contract_sha256=ch,model=config['validated_model'],
                        prompt_version=config['version'],raw_output=result.text,parsed=parsed,parse_errors=errors,
                        finish_reason=result.finish_reason,input_tokens=len(out.prompt_token_ids),
                        output_tokens=len(result.token_ids),output_token_ids=list(result.token_ids),batch_seconds=duration,
                        actual_batch_size=len(selected),engine_max_num_seqs=batch,generation_group_start=group[0][0]))
                offset=0
                for start,stop,path in group:
                    count=stop-start;save_rows(path,rr[offset:offset+count]);offset+=count;p.done+=count
                fresh=p.done-completed
                if fresh>=32:p.estimates[3]=len(docs)*(time.time()-p.stage_started)/fresh
        if retry_stop is not None:
            restore_batch(root,ch,a.batch_size)
            p.log(f'[recovery complete] Failed group saved; restore batch={a.batch_size} with a fresh engine. Completed={p.done}/{len(docs)}; remaining={len(docs)-p.done}')
            raise SystemExit(75)
        p.complete(f'All {len(docs):,} documents durably generated; {root}/generation_shards')
        p.start(5,len(docs),unit='document',estimate=90.)
        with heartbeat(p):summary=aggregate(root,docs,spans,ch,p)
        p.complete(str(root/'summary.json'));p.close(force_complete=True)
        print('[complete] '+json.dumps(summary['counts']))
    except BaseException as exc:
        if isinstance(exc,SystemExit) and exc.code==75:
            p.close();raise
        p.log(f'[FAILED] stage={p.index}/5 item={p.done}/{p.total}; {type(exc).__name__}: {exc}; durable={root}/generation_shards; rerun identical command to resume')
        p.close();raise


if __name__=='__main__':main()

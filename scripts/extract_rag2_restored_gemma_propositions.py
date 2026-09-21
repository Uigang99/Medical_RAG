#!/usr/bin/env python3
"""Full restored-union extraction with the frozen, pilot-tested Gemma prompt.

Reuses the tested renderer/parser/engine settings; batch shards are immutable.
No retrieval, quality-based filtering, prompt repair, or automatic GPU retries.
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
    result=dict(counts=counts,by_source=by_source,scope='Automatic completeness/format audit, not semantic accuracy; all 88,294 sources retained',
                contract_sha256=contract_hash)
    save(root/'summary.json',result)
    save(root/'COMPLETE.json',dict(contract_sha256=contract_hash,documents=len(docs)))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=INPUT_ROOT/'restored_chunks.jsonl')
    ap.add_argument('--output-dir',type=Path,default=OUTPUT)
    ap.add_argument('--max-model-len',type=int,default=16384,
                    help='Context capacity only; prompt and output budget remain frozen. Use a separate output directory if changed.')
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args();root=a.output_dir;root.mkdir(parents=True,exist_ok=True)
    # Different concurrent commands must not race over shards or aggregation.
    import fcntl
    lock=(root/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=TwoLineProgress(['validate source/cache','tokenize input budgets','load Gemma31B','generate propositions','write outputs'])
    p.plan(['88,294 fixed restored documents (includes 58 source-review cases)',
            'cached or validate all prompt token budgets; no truncation',
            'Gemma31B on GPU 1, context capacity=16384 (pilot8192); output remains4096; load ~25s',
            'batch=8, 8-document durable shards; measured pilot ~2.8–3.1s/document',
            'stream source-linked outputs and warnings'])
    p.estimates=[10.,None,25.,88294*3.1,90.]
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
            contract=dict(version='restored_union88294_gemma_faithful_v3',input_path=str(a.input.resolve()),
                input_sha256=file_sha(a.input),documents=len(docs),by_source=EXPECTED,
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
            ch=sha(json.dumps(contract,sort_keys=True))
            mf=root/'manifest.json'
            if mf.exists() and json.loads(mf.read_text())['contract_sha256']!=ch:
                raise RuntimeError('Output contract differs; choose a new output-dir before GPU loading')
            if not mf.exists():
                from extract_rag2_top8_standalone_propositions import git_commit
                save(mf,dict(contract=contract,contract_sha256=ch,git_commit=git_commit(),
                    runner_sha256=file_sha(Path(__file__)),hypothesis='Scale frozen extraction configuration without changing documents or prompt'))
            batch=g['batch_size'];spans=[];completed=0;missing=[]
            for start in range(0,len(docs),batch):
                stop=min(start+batch,len(docs));path=root/'generation_shards'/f'batch_{start:07d}_{stop:07d}.jsonl'
                spans.append((start,stop,path))
                if path.exists():shard_records(path,docs,start,stop,ch);completed+=stop-start
                else:missing.append((start,stop,path))
        p.complete(f'cached={completed:,}; remaining={len(docs)-completed:,}; output={root}')
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
            with heartbeat(p):
                llm=LLM(model=str(model),dtype=g['dtype'],tensor_parallel_size=1,
                    gpu_memory_utilization=g['gpu_memory_utilization'],max_model_len=g['max_model_len'],
                    max_num_seqs=batch,max_num_batched_tokens=4096,seed=g['seed'],
                    enforce_eager=True,enable_prefix_caching=True,language_model_only=True,disable_log_stats=True)
            sampling=SamplingParams(temperature=0.,max_tokens=g['max_output_tokens'],skip_special_tokens=False)
        p.complete('Gemma31B loaded' if missing else 'All shards cached; GPU loading skipped')
        p.start(4,len(docs),initial=completed,unit='document',estimate=len(docs)*3.1)
        (root/'generation_shards').mkdir(exist_ok=True)
        with heartbeat(p):
            for start,stop,path in missing:
                prompts=[render(tokenizer,prompt,d['restored_text']) for d in docs[start:stop]]
                tick=time.monotonic();generated=llm.generate(prompts,sampling,use_tqdm=False)
                duration=time.monotonic()-tick;rr=[]
                if len(generated)!=stop-start:raise RuntimeError('vLLM output count mismatch')
                for d,out in zip(docs[start:stop],generated):
                    result=out.outputs[0];parsed,errors=parse_model_output(clean_completion(result.text),d['restored_text'])
                    rr.append(dict(stable_id=d['stable_id'],source=d['source'],doc_id=d.get('doc_id'),
                        text_sha256=sha(d['restored_text']),contract_sha256=ch,model=config['validated_model'],
                        prompt_version=config['version'],raw_output=result.text,parsed=parsed,parse_errors=errors,
                        finish_reason=result.finish_reason,input_tokens=len(out.prompt_token_ids),
                        output_tokens=len(result.token_ids),output_token_ids=list(result.token_ids),batch_seconds=duration))
                save_rows(path,rr);p.done+=len(rr)
                fresh=p.done-completed
                if fresh>=32:p.estimates[3]=len(docs)*(time.time()-p.stage_started)/fresh
        p.complete(f'All {len(docs):,} documents durably generated; {root}/generation_shards')
        p.start(5,len(docs),unit='document',estimate=90.)
        with heartbeat(p):summary=aggregate(root,docs,spans,ch,p)
        p.complete(str(root/'summary.json'));p.close(force_complete=True)
        print('[complete] '+json.dumps(summary['counts']))
    except BaseException as exc:
        p.log(f'[FAILED] stage={p.index}/5 item={p.done}/{p.total}; {type(exc).__name__}: {exc}; durable={root}/generation_shards; rerun identical command to resume')
        p.close();raise


if __name__=='__main__':main()

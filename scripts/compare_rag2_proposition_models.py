#!/usr/bin/env python3
"""Paired 3-model x 2-prompt, document-only proposition quality pilot.

Automatic diagnostics are not semantic accuracy or information recall.
Each model runs in a fresh subprocess and exits before the next model loads.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from extract_rag2_top8_standalone_propositions import (
    SYSTEM_PROMPT, EXPECTED_SYSTEM_PROMPT_SHA256, TwoLineProgress,
    parse_model_output, validate_prompt_contract,
)

ROOT=Path(__file__).resolve().parents[1]
MODEL_NAMES=['Qwen3.5-9B','gemma-4-31B-it','Qwen3.5-27B']
FAITHFUL_PROMPT='''Convert the supplied passage into a set of independently understandable factual propositions using only what the passage states.

Preserve meaning before minimizing sentence length. A proposition may contain several clauses when they are necessary to express one complete claim. Keep the subject, action or relationship, and affected object correctly connected.
Keep each claim together with its necessary conditions, population, quantities, comparisons, time, negation, uncertainty, and study-specific scope. Repeat these details in separate propositions when needed; do not split a condition away from the claim it qualifies. Do not turn an association into causation, a side effect into an intended purpose, or a study observation into a universal fact.
Replace pronouns and vague references with concrete names or descriptive phrases only when the passage identifies the referent. Identify the study or patient descriptively when the passage allows it; do not rely on phrases like "the study" alone. Do not expand an abbreviation or repair an apparent factual error using outside knowledge.
Do not guess missing words at a damaged boundary. Omit an unrecoverable fragment, but retain the other supported claims. Do not invent statements from headings or disconnected table entries.
Cover all substantive claims, including relevant study methods and results. Avoid duplicate claims. Reword only as much as needed for clarity and independence; copying an already suitable source sentence is acceptable.
For each proposition, quote an exact contiguous passage span that supports it. The quote may span multiple sentences when needed to resolve a referent or preserve a condition. Do not shorten the quote at the expense of support.

Return JSON only: {"propositions":[{"claim":"standalone proposition","evidence":"exact source span"}]}'''
PROMPTS={'standalone_v2':SYSTEM_PROMPT,'faithful_units_v1':FAITHFUL_PROMPT}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def rows(path):
    with path.open() as f:
        for line in f:yield json.loads(line)


def save_rows(path,items):
    tmp=path.with_suffix('.jsonl.partial')
    with tmp.open('w') as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    tmp.replace(path)


def select_cohort(path):
    pools={s:{'short':[],'medium':[],'long':[],'warning':[]} for s in ['cpg','pmc','pubmed','textbooks']}
    for r in rows(path):
        n=len(r['restored_text'].split())
        if r['status']!='boundary_expanded_candidate' or n>600:continue
        kind='warning' if r['quality_flags'] else ('short' if n<=140 else 'medium' if n<=190 else 'long')
        pools[r['source']][kind].append(r)
    rng=random.Random(20260921);selected=[]
    for s in pools:
        for kind,pool in pools[s].items():
            if len(pool)<2:raise ValueError(f'Insufficient cohort stratum: {s}/{kind}')
            for r in rng.sample(sorted(pool,key=lambda x:x['stable_id']),2):
                selected.append(dict(stable_id=r['stable_id'],source=s,stratum=kind,
                    text=r['restored_text'],original_text=r['original_text'],
                    text_sha256=sha(r['restored_text']),quality_flags=r['quality_flags'],
                    context_chunk_ids=r['context_chunk_ids']))
    return selected


def render(tokenizer,prompt,text):
    return tokenizer.apply_chat_template([{'role':'system','content':prompt},
        {'role':'user','content':'SOURCE PASSAGE:\n'+text}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)


def clean_completion(raw):
    # Only remove known model envelope markers; never repair JSON or claims.
    text=raw.strip()
    for marker in ['<eos>','<|im_end|>','<|endoftext|>','<turn|>']:
        if text.endswith(marker):text=text[:-len(marker)].rstrip()
    if text.startswith('<|channel>final\n'):text=text[len('<|channel>final\n'):]
    return text


def ready_weights(path):
    index=path/'model.safetensors.index.json'
    if not index.exists():return False
    names=set(json.loads(index.read_text())['weight_map'].values())
    if not all((path/n).exists() and (path/n).stat().st_size>0 for n in names):return False
    manifest=path/'download_manifest.json'
    if path.name in MODEL_NAMES[1:]:
        if not manifest.exists():return False
        m=json.loads(manifest.read_text())
        if m.get('status')!='complete':return False
        return all((path/n).stat().st_size==size for n,size in m['files'].items())
    return True


def worker(args):
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
    os.environ.setdefault('VLLM_LOGGING_LEVEL','ERROR')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from vllm import LLM,SamplingParams
    root=args.output_root;model=ROOT.parent/'models'/args.worker
    manifest=json.loads((root/'manifest.json').read_text())
    g=manifest['generation'];out=root/args.worker;out.mkdir(exist_ok=True)
    tasks=list(rows(root/f'prompts_{args.worker}.jsonl'))
    cached={r['task_id']:r for r in rows(out/'outputs.jsonl')} if (out/'outputs.jsonl').exists() else {}
    save(out/'status.json',dict(stage='loading',completed=len(cached),total=len(tasks)))
    missing=[t for t in tasks if t['task_id'] not in cached]
    if not missing:
        save(out/'status.json',dict(stage='complete',completed=len(tasks),total=len(tasks)));return
    kw=dict(model=str(model),dtype='bfloat16',tensor_parallel_size=1,
            gpu_memory_utilization=g['gpu_memory_utilization'],max_model_len=g['max_model_len'],
            max_num_seqs=g['batch_size'],max_num_batched_tokens=4096,seed=g['seed'],
            enforce_eager=True,enable_prefix_caching=True,language_model_only=True,disable_log_stats=True)
    if args.worker.startswith('Qwen'):kw['additional_config']={'gdn_prefill_backend':'triton'}
    llm=LLM(**kw)
    save(out/'status.json',dict(stage='generating',completed=len(cached),total=len(tasks)))
    sampling=SamplingParams(temperature=0.0,max_tokens=g['max_output_tokens'],skip_special_tokens=False)
    for start in range(0,len(missing),g['batch_size']):
        batch=missing[start:start+g['batch_size']];tick=time.monotonic()
        result=llm.generate([t['rendered_prompt'] for t in batch],sampling,use_tqdm=False)
        duration=time.monotonic()-tick
        for t,res in zip(batch,result):
            completion=res.outputs[0];raw=completion.text
            parsed,errors=parse_model_output(clean_completion(raw),t['text'])
            record={k:v for k,v in t.items() if k!='rendered_prompt'}
            record.update(raw_output=raw,parsed=parsed,parse_errors=errors,
                finish_reason=completion.finish_reason,output_tokens=len(completion.token_ids),
                output_token_ids=list(completion.token_ids),batch_seconds=duration)
            cached[t['task_id']]=record
        save_rows(out/'outputs.jsonl',[cached[t['task_id']] for t in tasks if t['task_id'] in cached])
        save(out/'status.json',dict(stage='generating',completed=len(cached),total=len(tasks)))
    save(out/'status.json',dict(stage='complete',completed=len(cached),total=len(tasks)))


def aggregate(root,cohort,models):
    all_rows=[];summary={}
    for model in models:
        records=list(rows(root/model/'outputs.jsonl'));all_rows.extend(records)
        for prompt in PROMPTS:
            rr=[r for r in records if r['prompt_version']==prompt]
            pp=[p for r in rr for p in (r['parsed'] or {}).get('propositions',[])]
            summary[f'{model}/{prompt}']=dict(documents=len(rr),
                valid_nonempty_documents=sum(bool((r['parsed'] or {}).get('propositions')) for r in rr),
                truncated_documents=sum(r['finish_reason']=='length' for r in rr),
                propositions=len(pp),exact_evidence=sum(p['evidence_exact_match'] for p in pp),
                context_warning_propositions=sum(p['decontextualization_flag'] for p in pp),
                semantic_precision=None,source_information_recall=None)
    save(root/'summary.json',dict(conditions=summary,
        interpretation='Automatic diagnostics only; no semantic accuracy or completeness claim. 24 ordinary + 8 flagged documents, balanced across four corpora; not population prevalence.'))
    save_rows(root/'comparison.jsonl',[dict(document=d,outputs=[r for r in all_rows if r['stable_id']==d['stable_id']]) for d in cohort])
    audit=[]
    for r in all_rows:
        for i,p in enumerate((r['parsed'] or {}).get('propositions',[])):
            audit.append(dict(task_id=r['task_id'],proposition_index=i,claim=p['claim'],evidence=p['evidence'],
                source_supported=None,subject_object_correct=None,conditions_preserved=None,
                standalone=None,issue_note=''))
    save_rows(root/'semantic_review_template.jsonl',audit)
    save_rows(root/'document_coverage_review_template.jsonl',[
        dict(task_id=r['task_id'],missing_source_claims=[],source_claim_count=None,covered_source_claim_count=None,
             review_status='not_reviewed') for r in all_rows])
    lines=['# Paired proposition comparison','',
        'Review source fidelity, argument roles, scope, independence, and omissions. Blank audit fields are NOT passing grades.','']
    for d in cohort:
        lines += [f"## {d['stable_id']} ({d['source']}, {d['stratum']})",'',d['text'],'']
        for r in all_rows:
            if r['stable_id']!=d['stable_id']:continue
            lines += [f"### {r['model']} / {r['prompt_version']}",'',f"Finish: {r['finish_reason']}; errors: {r['parse_errors']}",'']
            if r['parsed']:
                for i,p in enumerate(r['parsed']['propositions'],1):
                    lines += [f"{i}. {p['claim']}",f"   Evidence: {p['evidence']}",'']
            else:lines += [r['raw_output'],'']
    (root/'comparison.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=ROOT/'datasets/kg/rag2_test6545_top1_2_4_8_16_sentence_restored_v1/restored_chunks.jsonl')
    ap.add_argument('--output-root',type=Path,default=ROOT/'results/rag2_restored_proposition_models32_v1')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--worker',choices=MODEL_NAMES)
    args=ap.parse_args()
    if args.worker:return worker(args)
    if not args.prepare_only and os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=1; this pilot uses only GPU 1')
    validate_prompt_contract()
    from transformers import AutoTokenizer
    root=args.output_root;root.mkdir(exist_ok=True,parents=True)
    cohort=select_cohort(args.input)
    generation=dict(dtype='bfloat16',thinking=False,temperature=0.0,max_output_tokens=4096,
        max_model_len=8192,batch_size=8,gpu_memory_utilization=0.80,seed=20260921,
        enforce_eager=True,language_model_only=True)
    tasks={};identities={};available={}
    for name in MODEL_NAMES:
        path=ROOT.parent/'models'/name
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        identities[name]=dict(path=str(path),config_sha256=file_sha(path/'config.json'),
            tokenizer_sha256=file_sha(path/'tokenizer.json'),chat_template_sha256=file_sha(path/'chat_template.jinja'))
        available[name]=ready_weights(path)
        tasks[name]=[]
        for d in cohort:
            for key,prompt in PROMPTS.items():
                rendered=render(tokenizer,prompt,d['text']);tokens=tokenizer.encode(rendered,add_special_tokens=False)
                if len(tokens)+generation['max_output_tokens']>generation['max_model_len']:
                    raise ValueError(f'Input/output budget exceeds context, no truncation allowed: {name}/{d["stable_id"]}')
                tasks[name].append(dict(task_id=sha(name+'|'+key+'|'+d['stable_id']),
                    stable_id=d['stable_id'],source=d['source'],stratum=d['stratum'],text=d['text'],
                    model=name,prompt_version=key,rendered_prompt=rendered,input_tokens=len(tokens)))
    manifest=dict(version='restored_32doc_3model_2prompt_v1',input_path=str(args.input),
        input_sha256=file_sha(args.input),cohort_sha256=sha(json.dumps(cohort,sort_keys=True)),
        model_identities=identities,prompts=PROMPTS,generation=generation,seed=20260921,
        selection='Each corpus: 2 <=140 word, 2 141-190 word, 2 191-600 word, 2 surface-warning documents. Unresolved restoration excluded.',
        hypothesis='Compare source-faithful proposition sets across model and prompt on identical restored source text',
        uses_questions_answers_labels=False,semantic_review='Manual source-relative review required; no LLM judge',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=file_sha(Path(__file__)),
        environment={n:importlib.metadata.version(n) for n in ['vllm','transformers','torch']})
    mf=root/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:
        raise RuntimeError('Pilot contract changed; use a new output root')
    save(mf,manifest);save_rows(root/'cohort.jsonl',cohort)
    for name in MODEL_NAMES:save_rows(root/f'prompts_{name}.jsonl',tasks[name])
    save(root/'preflight.json',dict(documents=len(cohort),conditions=6,outputs=192,
        weights_ready=available,max_input_tokens={n:max(t['input_tokens'] for t in ts) for n,ts in tasks.items()},
        gpu_execution_tested=False))
    if args.prepare_only:
        print(json.dumps(json.loads((root/'preflight.json').read_text()),indent=2));return
    if not all(available.values()):raise RuntimeError(f'Model downloads incomplete: {available}; no GPU loaded')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no generation started')
    names=['prepare']+[s for n in MODEL_NAMES for s in [f'load {n}',f'generate {n}']]+['aggregate']
    p=TwoLineProgress(names);p.plan(['cached cohort/tokenizer preflight']+[f'{n}: 64 outputs, load once' for n in MODEL_NAMES]+['source-relative review'])
    p.start(1,1);p.done=1;p.complete('32 shared documents; six conditions')
    try:
        for j,name in enumerate(MODEL_NAMES):
            out=root/name;out.mkdir(exist_ok=True);log=out/'worker.log'
            cached=list(rows(out/'outputs.jsonl')) if (out/'outputs.jsonl').exists() else []
            expected={t['task_id'] for t in tasks[name]}
            if len({r['task_id'] for r in cached})!=len(cached) or any(r['task_id'] not in expected for r in cached):
                raise ValueError(f'Invalid cached task identities: {name}')
            p.start(2+2*j,1,unit='model')
            if len(cached)==len(expected):
                p.complete('model load skipped: all outputs cached');p.start(3+2*j,len(expected),initial=len(expected));p.complete('cached');continue
            save(out/'status.json',dict(stage='loading',completed=len(cached),total=len(expected)))
            with log.open('a') as handle:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',name,'--output-root',str(root)],stdout=handle,stderr=subprocess.STDOUT)
                generating=False
                while True:
                    state=json.loads((out/'status.json').read_text())
                    if state['stage']!='loading' and not generating:
                        p.complete(f'loaded; details={log}');p.start(3+2*j,len(expected),initial=len(cached),unit='output');generating=True
                    if generating:
                        p.done=state['completed']
                        fresh=p.done-len(cached)
                        if fresh>0:p.estimates[p.index-1]=len(expected)*(time.time()-p.stage_started)/fresh
                    p.render()
                    code=proc.poll()
                    if code is not None:
                        if code:raise RuntimeError(f'Worker failed: {name}, exit={code}, log={log}; durable outputs retained')
                        break
                    time.sleep(2)
                if not generating:
                    p.complete('model worker finished');p.start(3+2*j,len(expected),unit='output')
                result=list(rows(out/'outputs.jsonl'))
                if {r['task_id'] for r in result}!=expected:raise RuntimeError(f'Missing outputs: {name}')
                p.complete(f'64 outputs saved: {out}')
        p.start(8,1,unit='report');aggregate(root,cohort,MODEL_NAMES);p.complete(str(root/'comparison.md'));p.close(force_complete=True)
        print(f'[complete] 192 outputs. Automatic diagnostics are not semantic quality scores: {root}')
    except BaseException:
        if 'proc' in locals() and proc.poll() is None:
            proc.terminate();proc.wait()
        p.log(f'[FAILED] stage={p.index}/{len(names)}; cached outputs retained at {root}; rerun identical command to resume')
        p.close();raise


if __name__=='__main__':main()

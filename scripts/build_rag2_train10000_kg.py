#!/usr/bin/env python3
"""Build official Train10k Top8 proposition/KG data with the frozen Test method.

No QA model training or evaluation. --preflight-only never loads a model or
scans the large retrieval JSONL. Default execution performs the entire build.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from medrag import train_kg_data as data
from medrag.training.graph_input_progress import Progress

PYTHON=ROOT.parent/'.venv_vllm/bin/python'
KG_PYTHON=ROOT.parent/'.venv_rag2_kg_mvp_py312/bin/python'
WORKER=ROOT/'scripts/run_rag2_train_kg_stage.py'
DEFAULT_ROOT=ROOT/'datasets/kg/rag2_official_train10000_top8_frozen_v1'
PHASES=['validate inputs/cache','tokenize prompt budgets','load model','generate durable shards','aggregate']
JOBS=[('cohort',None,PYTHON,['select official Train / split']),
      ('retrieval',None,KG_PYTHON,['reuse balanced Top8 / resolve corpus text']),
      ('restore',None,KG_PYTHON,['restore sentence boundaries']),
      ('propositions',None,PYTHON,PHASES),('repair',None,PYTHON,PHASES),
      ('usable',None,PYTHON,['select structurally usable propositions']),
      ('step2',None,PYTHON,PHASES),('step3',None,PYTHON,PHASES),('step4',None,PYTHON,PHASES),
      ('step5','inventory',KG_PYTHON,['inventory endpoints']),
      ('step5','candidates',KG_PYTHON,['local UMLS candidates']),
      ('step5','prepare',KG_PYTHON,['prepare identity tasks']),
      ('step5','verify',PYTHON,['Qwen contextual identity verification']),
      ('step5','aggregate',KG_PYTHON,['annotate frozen frames']),
      ('kg',None,PYTHON,['validate Stage5','materialize SQLite','export provenance','connectivity audit','integrity checks']),
      ('export',None,PYTHON,['export question-local Top8 graph/text mappings'])]
DEPENDENCIES=['scripts/build_rag2_train10000_kg.py','scripts/run_rag2_train_kg_stage.py','medrag/train_kg_data.py',
 'scripts/restore_rag2_retrieved_sentence_boundaries.py','scripts/extract_rag2_restored_gemma_propositions.py',
 'scripts/retry_rag2_restored_top8_gemma_propositions.py','scripts/prepare_rag2_gemma_union_usable_propositions.py',
 'scripts/extract_rag2_stage2_relation_discovery_full.py','scripts/extract_rag2_stage2_relation_discovery_union_full.py',
 'scripts/extract_rag2_stage3_endpoint_binding_union_full.py','scripts/filter_rag2_stage4_risk_routed_entailment_union_full.py',
 'scripts/normalize_rag2_stage5_contextual_union_full.py','scripts/build_rag2_stage6_7_surface_kg_union_full.py',
 'scripts/compare_rag2_proposition_models.py','scripts/extract_rag2_top8_standalone_propositions.py',
 'scripts/pilot_rag2_assertion_then_binding_v3.py','scripts/evaluate_rag2_stage3_direct_endpoint_binding_document_cohort.py',
 'scripts/evaluate_rag2_stage3_edge_emission_gate.py','scripts/evaluate_rag2_stage4_frame_entailment_filter.py',
 'scripts/evaluate_rag2_stage4_risk_routed_entailment.py','scripts/evaluate_rag2_stage4_surface_frame_validation.py',
 'scripts/pilot_rag2_stage5_contextual_entity_generalization.py','scripts/normalize_rag2_bound_frame_entities.py',
 'prompts/proposition_standalone_faithful_v3.config.json','prompts/proposition_standalone_faithful_v3.txt']


def preflight(root):
    for p in [PYTHON,KG_PYTHON,*(ROOT/x for x in DEPENDENCIES),data.BENCH/'all/test.jsonl']:
        if not p.is_file(): raise FileNotFoundError(p)
    # Read metadata only, not multi-GB retrieval data.
    cache={}
    for ds in data.COUNTS:
        data.verify_retrieval(data.load(data.cache_manifest(ds)))
        for p in [data.BENCH/ds/'train.jsonl',data.CACHE/ds/'train/candidates_top8.jsonl']:
            if not p.is_file(): raise FileNotFoundError(p)
        p=data.CACHE/ds/'train/candidates_top8.jsonl'
        cache[ds]={'manifest_sha256':data.sha_file(data.cache_manifest(ds)),
                   'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    cfg=data.load(ROOT/'prompts/proposition_standalone_faithful_v3.config.json')
    if data.sha_file(ROOT/'prompts'/cfg['system_prompt_file'])!=cfg['system_prompt_file_sha256']:
        raise RuntimeError('Frozen proposition prompt hash mismatch')
    models={}
    for name in ('gemma-4-31B-it','Qwen3.5-27B'):
        path=ROOT.parent/'models'/name
        index=data.load(path/'model.safetensors.index.json')
        missing=[f for f in set(index['weight_map'].values()) if not (path/f).is_file()]
        if missing: raise RuntimeError(f'Checkpoint shards missing for {name}: {missing}')
        models[name]={'path':str(path),'config_sha256':data.sha_file(path/'config.json')}
    if not (ROOT.parent/'models/RAG2-Medical-KG-MVP/scispacy_cache/datasets').is_dir():
        raise FileNotFoundError('Local UMLS cache missing')
    for source in data.SOURCES:
        if not (ROOT/'databases/vector_db/RAG_Square'/source/'manifest.json').is_file():
            raise FileNotFoundError(f'Indexed corpus missing: {source}')
    subprocess.run([str(KG_PYTHON),'-c',
        "import sys;sys.path[:0]=['.','scripts'];import pysbd,numpy;import normalize_rag2_stage5_contextual_union_full"],
        cwd=ROOT,check=True,capture_output=True,text=True)
    free=shutil.disk_usage(root if root.exists() else root.parent).free/2**30
    if free<80: raise RuntimeError(f'Only {free:.1f} GiB free; reserve at least 80 GiB')
    return {'version':data.VERSION,'hypothesis':'Transfer frozen Test source-processing method to independent official Train sources',
        'cohort':{'medmcqa':7000,'medqa':3000,'train':9000,'validation':1000,'seed':data.SEED},
        'retrieval':cache,'models':models,'top_k':8,'document_order':'cached cross-encoder rank',
        'qa_labels_used_by_KG':False,'source_only_generation':True,
        'source_artifact_reuse':'reuse retrieval only; no old Qwen propositions or Test generation shards',
        'code':{f:data.sha_file(ROOT/f) for f in DEPENDENCIES},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope_limits':['No semantic quality claim beyond frozen pilot',
            'Same source documents may occur in Train/Val/Test; QA labels and example selection remain separate',
            'Question views contain only edges sourced by that question Top8, not global neighbours'],
        'output_root':str(root.resolve())}


def run_job(args,job,offset,progress,env):
    stage,mode,python,names=job
    slug=stage+('_'+mode if mode else '')
    marker=args.output_root/'workflow'/f'{slug}.json'
    if marker.exists():
        prior=data.load(marker)
        if prior['workflow_contract_sha256']!=args.contract_hash: raise RuntimeError('Workflow marker contract mismatch')
        for i,name in enumerate(names):
            progress.stage(offset+i,1,initial=1)
            progress.complete(f'cached {slug}')
        return
    cmd=[str(python),'-u',str(WORKER),'--root',str(args.output_root),'--stage',stage,'--profile',args.profile]
    if mode: cmd+=['--mode',mode]
    log=args.output_root/'logs'/f'{slug}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    started=time.time(); attempt=0
    progress.stage(offset,1)
    while True:
        attempt+=1; last_phase=None; oom=False; previous=[]
        if args.profile=='h200-throughput' and (stage in ('propositions','repair','step2','step3','step4') or (stage=='step5' and mode=='verify')):
            gpu=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used',
                '--format=csv,noheader,nounits'],capture_output=True,text=True)
            if gpu.returncode:
                raise RuntimeError('Cannot inspect GPU 1; no model launched. '+gpu.stderr.strip())
            total,used=map(float,gpu.stdout.strip().split(','))
            if total-used < total*0.90+1024:
                raise RuntimeError(f'GPU 1 has only {total-used:.0f}/{total:.0f} MiB free; throughput profile requires 90% plus 1 GiB headroom. Stop the other GPU job before resuming.')
        with log.open('a') as f:
            f.write(f'\nATTEMPT {attempt} time={time.time()} cmd={cmd}\n'); f.flush()
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in proc.stdout:
                    f.write(line); f.flush()
                    lower=line.lower()
                    oom |= 'cuda out of memory' in lower or 'torch.outofmemoryerror' in lower
                    previous=(previous+[line.rstrip()])[-12:]
                    if line.startswith('@@TRAIN_KG '):
                        e=json.loads(line[len('@@TRAIN_KG '):]); phase=int(e['phase'])
                        if e.get('complete'):
                            progress.complete(e.get('detail','')); continue
                        if phase<1 or phase>len(names): raise RuntimeError('Invalid child progress phase')
                        key=(phase,e['total'],e.get('initial',0))
                        if key!=last_phase:
                            progress.names[offset+phase-1]=f'{slug}: {e.get("name",names[phase-1])}'
                            progress.stage(offset+phase-1,e['total'],initial=e.get('initial',0),unit=e.get('unit','item'))
                            last_phase=key
                        if e['done']>=progress.done: progress.update(e['done']-progress.done)
                    elif 'warning' in lower or 'error' in lower or '[oom' in lower:
                        progress.log(line.rstrip())
                status=proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(); raise
        if status==0: break
        if status==75 and stage in ('propositions','repair','step2','step3','step4','step5'):
            progress.log(f'[{slug}] controlled durable-shard resume; details={log}'); continue
        if oom and stage in ('propositions','repair','step2','step3','step4'):
            with log.open('a') as f:
                retry=subprocess.run(cmd+['--reduce-batch-after-oom'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if retry.returncode==0:
                progress.log(f'[{slug}] confirmed CUDA OOM: retry current shard at lower batch, then restore'); continue
        for line in previous: progress.log(line)
        raise RuntimeError(f'{slug} failed (exit {status}); durable output={args.output_root}; log={log}. Rerun identical command after correction.')
    for i in range((last_phase[0] if last_phase else 1)-1,len(names)):
        if progress.index!=offset+i: progress.stage(offset+i,1,initial=1)
        progress.complete(f'{slug}; log={log}')
    data.write_json(marker,{'workflow_contract_sha256':args.contract_hash,'elapsed_seconds':time.time()-started,'log':str(log)})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root',type=Path)
    ap.add_argument('--profile',choices=('reference','h200-throughput'),default='reference')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--stop-after',choices=['cohort','retrieval','restore','propositions','repair','usable','step2','step3','step4','step5','kg','export'],default='export')
    args=ap.parse_args()
    if args.output_root is None:
        args.output_root=DEFAULT_ROOT if args.profile=='reference' else DEFAULT_ROOT.with_name('rag2_official_train10000_top8_h200_throughput_v2')
    contract=preflight(args.output_root)
    contract['execution_profile']=args.profile
    contract['execution_note']='No measured speedup yet; batch-dependent floating-point differences can change outputs. Prompts/parsers/output limits are unchanged.'
    if args.preflight_only:
        print(json.dumps({'status':'preflight_ok','questions':10000,'profile':args.profile,'output_root':str(args.output_root),'models':contract['models'],
            'GPU_loaded':False,'large_cache_scanned':False,'outputs_written':False},ensure_ascii=False,indent=2))
        return
    args.output_root.mkdir(parents=True,exist_ok=True)
    lock=(args.output_root/'.workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resume_whitespace_fix(args.output_root,contract)
    args.contract_hash=data.freeze(args.output_root,contract)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             VLLM_LOGGING_LEVEL='ERROR',VLLM_USE_FLASHINFER_SAMPLER='0',PYTHONUNBUFFERED='1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    selected=[]
    for job in JOBS:
        selected.append(job)
        if job[0]==args.stop_after and (job[0]!='step5' or job[1]=='aggregate'): break
    names=[f'{s}{"/"+m if m else ""}: {n}' for s,m,_,nn in selected for n in nn]
    p=Progress(names); offset=0
    p.log('[plan] official Train 7000 MedMCQA + 3000 MedQA -> 9000/1000; cached balanced Top8 -> restored source -> frozen Gemma proposition/repair -> frozen Step2/3 -> Qwen Step4/5 -> CPU KG; GPU 1 only. Overall ETA calibrating; unknown document count until retrieval dedup. No QA training/evaluation.')
    try:
        for job in selected:
            run_job(args,job,offset,p,env);offset+=len(job[3])
        p.log(f'[finished] stop_after={args.stop_after}; output={args.output_root}')
    except BaseException as e:
        p.log(f'[FAILED] stage={p.index+1}/{len(names)} item={p.done}/{p.total}; {e}')
        raise
    finally: p.close()


def resume_whitespace_fix(root,contract):
    """One known pre-GPU failure only; never relax a generated-data contract."""
    path=root/'manifest.json'
    if not path.exists(): return
    old=data.load(path)
    if old['contract']==contract: return
    known='5f4a8b6a385057859fde5bc2633a2ab672959f3253f83f8ff57bb32fbf9fcdb6'
    if old.get('contract_sha256')!=known: return
    if data.digest(old['contract'])!=known:
        raise RuntimeError('Original workflow contract hash is invalid')
    if any((root/s).exists() for s in ('retrieval','restored','propositions','repair','usable','step2','step3','step4','step5','kg')):
        raise RuntimeError('Whitespace migration permitted only before retrieval artifacts/GPU work')
    if {p.name for p in (root/'workflow').glob('*.json')}!={'cohort.json'}:
        raise RuntimeError('Whitespace migration requires only the completed cohort stage')
    before=json.loads(json.dumps(old['contract']))
    after=json.loads(json.dumps(contract))
    for value in (before,after):
        value.pop('git_commit',None)
        for f in ('medrag/train_kg_data.py','scripts/build_rag2_train10000_kg.py'):
            value['code'].pop(f)
    if before!=after:
        raise RuntimeError('Inputs/settings changed beyond whitespace fix; cannot migrate')
    archive=root/'manifest.before_whitespace_fix.json'
    if archive.exists() and data.load(archive)!=old:
        raise RuntimeError('Whitespace migration archive differs')
    data.write_json(archive,old)
    marker_path=root/'workflow/cohort.json';marker=data.load(marker_path)
    new_hash=data.digest(contract)
    if marker['workflow_contract_sha256'] not in (known,new_hash):
        raise RuntimeError('Unexpected cohort completion contract')
    marker['workflow_contract_sha256']=new_hash
    marker['previous_workflow_contract_sha256']=known
    data.write_json(marker_path,marker)
    data.write_json(path,{'contract':contract,'contract_sha256':new_hash})
    print('[resume fix] Whitespace-only comparison; original manifest archived; existing cohort preserved; no GPU artifacts changed.',flush=True)


if __name__=='__main__':main()

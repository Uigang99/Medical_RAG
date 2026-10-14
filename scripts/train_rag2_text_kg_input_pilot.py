#!/usr/bin/env python3
"""Train TEXT or KG input adapters on a grouped repartition of TEST6545.

No held-out answer generation or final evaluation exists in this entry point.
Validation is teacher-forced answer loss for checkpoint selection only.
--preflight-only materializes/checks CPU train/validation data without loading
LLM weights, downloading an encoder, computing embeddings, or allocating GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from medrag.training.graph_input_data import (  # noqa: E402
    VERSION, SYSTEM, SourceGraph, atomic_json, atomic_jsonl, digest, ensure_contract,
    identity, jsonl, load_population, pack_pair, sha_file, split_questions,
)
from medrag.training.graph_input_progress import Progress  # noqa: E402
from medrag.training.graph_input_modules import training_attention_backend  # noqa: E402

KG_ROOT = ROOT / "datasets/kg/rag2_test6545_union88294_stage7_surface_kg_v1"
ENCODER_REPO = "sentence-transformers/all-roberta-large-v1"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("text", "graph"), required=True)
    p.add_argument("--model", type=Path, default=ROOT.parent / "models/Llama-3-8B-Instruct")
    p.add_argument("--encoder-model", type=Path, default=ROOT.parent / "models/all-roberta-large-v1")
    p.add_argument("--download-encoder", action="store_true", help="Explicitly allow first-run public SBERT download")
    p.add_argument("--benchmark-root", type=Path, default=ROOT / "datasets/benchmark")
    p.add_argument("--candidate-cache", type=Path, default=ROOT / "databases/run_cache/rag2_pced_semantic_labeled_dynamic_topk_v2/top8/candidates.jsonl")
    p.add_argument("--graph", type=Path, default=KG_ROOT / "graph.sqlite3")
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets/pilots/rag2_text_kg_input_test_resplit_v1")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/rag2_text_kg_input_training_v1_sdpa_fix")
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--validation-size", type=int, default=256)
    p.add_argument("--pilot-evaluation-size", type=int, default=512, help="Reserve IDs only; never evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--effective-batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--text-soft-tokens", type=int, default=10)
    p.add_argument("--graph-hidden", type=int, default=1024)
    p.add_argument("--graph-layers", type=int, default=4)
    p.add_argument("--graph-heads", type=int, default=4)
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--checkpoint-steps", type=int, default=25)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers")}


def bundle_identity(path):
    files = sorted(set(list(path.glob("*.safetensors")) + list(path.glob("*.bin")) +
                       [p for p in path.rglob("*.json") if ".cache" not in p.parts] +
                       list(path.glob("*.txt"))))
    if not files or not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise FileNotFoundError(f"Incomplete model checkpoint: {path}")
    return [identity(p) for p in files]


def check_paths(args):
    if min(args.train_size, args.validation_size, args.micro_batch_size, args.effective_batch_size,
           args.epochs, args.checkpoint_steps, args.embedding_batch_size, args.text_soft_tokens) < 1:
        raise ValueError("Training sizes and batch settings must be positive")
    if args.graph_layers < 1 or args.graph_heads < 1 or args.graph_hidden < 1 or args.graph_hidden % args.graph_heads:
        raise ValueError("Invalid graph architecture")
    if args.train_size + args.validation_size + args.pilot_evaluation_size > 6545:
        raise ValueError("Split sizes exceed TEST6545")
    if args.micro_batch_size > args.effective_batch_size:
        raise ValueError("micro batch exceeds effective batch")
    for p in (args.model / "config.json", args.model / "tokenizer_config.json",
              args.graph, args.graph.parent / "manifest.json", args.graph.parent / "summary.json",
              args.candidate_cache):
        if not p.is_file():
            raise FileNotFoundError(p)
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("model_type") != "llama" or args.max_context > model_config["max_position_embeddings"]:
        raise ValueError("Expected Llama within its native context length")
    kg = json.loads((args.graph.parent / "manifest.json").read_text())
    if kg["contract"]["scope"].get("qa_labels_used") is not False:
        raise ValueError("KG provenance must explicitly exclude QA labels")
    if any(args.output_root.resolve() == p.resolve() or args.data_root.resolve() == p.resolve()
           for p in (args.model, args.graph.parent, args.benchmark_root)):
        raise ValueError("Output cannot overwrite an input directory")
    if shutil.disk_usage(ROOT).free < 8 * 1024 ** 3:
        raise RuntimeError("Need at least 8 GiB free for caches and atomic adapter checkpoints")
    return kg, bundle_identity(args.model)


@contextlib.contextmanager
def lock_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}; do not run both arms on GPU 1 concurrently") from exc
        yield


def prepare_data(args, tokenizer, progress):
    root = args.data_root
    population, benchmark_ids = load_population(args.benchmark_root)
    contract = {
        "version": VERSION, "original_split": "test", "development_only": True,
        "seed": args.seed, "counts": {"train": args.train_size, "validation": args.validation_size,
           "pilot_evaluation": args.pilot_evaluation_size,
           "reserved": 6545-args.train_size-args.validation_size-args.pilot_evaluation_size},
        "benchmark": benchmark_ids, "candidates": identity(args.candidate_cache),
        "graph": identity(args.graph), "graph_manifest": identity(args.graph.parent / "manifest.json", full_hash=True),
        "graph_summary": identity(args.graph.parent / "summary.json", full_hash=True),
        "tokenizer": [identity(args.model / n, full_hash=True) for n in ("tokenizer.json", "tokenizer_config.json")],
        "max_context": args.max_context, "text_soft_tokens": args.text_soft_tokens,
        "system_prompt": SYSTEM, "target": "Answer: gold option letter + Llama eot; NO rationale",
        "selection": "dataset-proportional, exact/near stem groups; no outcomes or graph connectivity",
        "packing": "remove whole trailing source propositions and their edges jointly until TEXT fits",
        "empty_graph": "retain question with zero graph representation / empty TEXT evidence",
        "top_k": 8, "extra_neighbors": False, "canonical_identity": "existing Stage5 entity_node_id only",
        "code": identity(ROOT / "medrag/training/graph_input_data.py", full_hash=True),
        "transformers_version": importlib.metadata.version("transformers"),
        "heldout_policy": "pilot_evaluation/reserved IDs only; no examples, targets, or inference in this trainer",
    }
    with lock_file(root / ".prepare.lock"):
        data_hash = ensure_contract(root / "manifest.json", contract)
        complete = root / "COMPLETE.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            if saved["contract_sha256"] != data_hash:
                raise RuntimeError("Prepared-data completion hash mismatch")
            for name, expected in saved["file_sha256"].items():
                if sha_file(root / name) != expected:
                    raise RuntimeError(f"Prepared data modified: {name}")
            records = list(jsonl(root / "train_validation.jsonl"))
            progress.stage(0, len(records), initial=len(records), unit="q")
            progress.complete(f"CPU cache reused: {root}; held-out evidence not loaded")
            return data_hash, records, saved
        assignments, report = split_questions(population, contract["counts"], args.seed)
        by_split = {r["sample_id"]: r["split"] for r in assignments}
        selected = {r["sample_id"]: r for r in population if by_split[r["sample_id"]] in ("train", "validation")}
        atomic_jsonl(root / "split_assignments.jsonl", assignments)
        atomic_json(root / "split_summary.json", report)
        candidates, seen = {}, set()
        all_ids = {r["sample_id"] for r in population}
        for row in jsonl(args.candidate_cache):
            sid = row["sample_id"]
            if sid in seen or sid not in all_ids:
                raise RuntimeError(f"Invalid candidate ID: {sid}")
            seen.add(sid)
            if sid not in selected:
                continue
            sample = selected[sid]
            if row["dataset"] != sample["dataset"] or row["row_idx"] != sample["row_idx"]:
                raise RuntimeError(f"Candidate alignment failed: {sid}")
            if row["key"] != f"{sample['dataset']}::test::{sid}::{sample['row_idx']}":
                raise RuntimeError(f"Candidate split mismatch: {sid}")
            docs = row["reranked_documents"]
            if len(docs) != 8 or [d["rerank_rank"] for d in docs] != list(range(1, 9)):
                raise RuntimeError(f"Not ordered Top-8: {sid}")
            ids = [d.get("stable_id") or d.get("corpus_id") for d in docs]
            if None in ids or len(set(ids)) != 8:
                raise RuntimeError(f"Invalid document identity: {sid}")
            candidates[sid] = ids
        if seen != all_ids or set(candidates) != set(selected):
            raise RuntimeError("Incomplete fixed TEST6545 candidate cache")
        ordered = sorted(selected, key=lambda sid: digest([args.seed, sid]))
        progress.stage(0, len(ordered), unit="q")
        source = SourceGraph(args.graph)
        records = []
        try:
            for start in range(0, len(ordered), 32):
                ids = ordered[start:start+32]
                path = root / "shards" / f"{start//32:05d}.json"
                if path.exists():
                    shard = json.loads(path.read_text())
                    if shard["contract_sha256"] != data_hash or shard["ids"] != ids:
                        raise RuntimeError(f"Incompatible data shard: {path}")
                else:
                    batch = []
                    for sid in ids:
                        row = {**selected[sid], "split": by_split[sid], "document_ids": candidates[sid]}
                        batch.append(pack_pair(tokenizer, row, source.collect(candidates[sid]),
                                               args.max_context, args.text_soft_tokens))
                    shard = {"contract_sha256": data_hash, "ids": ids, "rows": batch}
                    atomic_json(path, shard)
                records.extend(shard["rows"])
                progress.update(len(ids))
        finally:
            source.close()
        for split in ("train", "validation"):
            if not any(r["split"] == split for r in records):
                raise RuntimeError(f"Empty {split} after grouped split")
        atomic_jsonl(root / "train_validation.jsonl", records)
        stats = {s: {"questions": sum(r["split"] == s for r in records),
                     "empty_graphs": sum(not r["graph"]["node_ids"] for r in records if r["split"] == s),
                     "directed_two_hop": sum(r["graph_stats"]["directed_two_hop"] for r in records if r["split"] == s),
                     "packed_questions": sum(r["packing"]["removed_propositions"] > 0 for r in records if r["split"] == s),
                     "triple_count": sum(r["graph_stats"]["edges"] for r in records if r["split"] == s)}
                 for s in ("train", "validation")}
        saved = {"contract_sha256": data_hash, "statistics": stats, "evaluation_executed": False,
                 "file_sha256": {name: sha_file(root / name) for name in
                    ("train_validation.jsonl", "split_assignments.jsonl", "split_summary.json")}}
        atomic_json(complete, saved)
        progress.complete(f"paired train/validation examples={len(records)}; cache={root}")
        return data_hash, records, saved


def torch_save(path, obj):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_torch(path, *, trusted=False):
    import torch
    return torch.load(path, map_location="cpu", weights_only=not trusted)


def ensure_encoder(args, progress):
    path = args.encoder_model
    provenance = path / "download_contract.json"
    complete = path / "download_complete.json"
    # A interrupted snapshot can already contain modules.json, but not weights.
    ready = ((path / "modules.json").is_file() and
             (any(path.glob("*.safetensors")) or any(path.glob("*.bin"))))
    if not ready or (provenance.exists() and not complete.exists()):
        if not args.download_encoder:
            raise FileNotFoundError(f"Sentence encoder missing: {path}; rerun graph wrapper with --download-encoder")
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        if provenance.exists():
            download = json.loads(provenance.read_text())
        else:
            info = HfApi().model_info(ENCODER_REPO)
            files = [s.rfilename for s in info.siblings]
            weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
            download = {"repo": ENCODER_REPO, "revision": info.sha, "weight_file": weights}
            atomic_json(provenance, download)
        progress.log(f"[dependency] downloading public {download['repo']} revision={download['revision']}; no QA data uploaded")
        snapshot_download(repo_id=download["repo"], revision=download["revision"], local_dir=str(path),
                          allow_patterns=[download["weight_file"], "*.json", "*.txt", "*.model", "*.bpe"])
        if not (path / "modules.json").is_file() or not (path / download["weight_file"]).is_file():
            raise RuntimeError(f"Sentence encoder snapshot incomplete: {path}")
        atomic_json(complete, download)
    return bundle_identity(path)


def prepare_features(args, rows, data_hash, progress):
    import torch
    if args.arm == "text":
        progress.stage(1, 1, initial=1)
        progress.complete("TEXT: no sentence encoder or graph embeddings required")
        return None, None, None
    progress.stage(1, 1, unit="encoder")
    encoder_identity = ensure_encoder(args, progress)
    texts = sorted({t for r in rows for t in r["graph"]["node_texts"] + r["graph"]["edge_texts"]})
    contract = {"data_contract": data_hash, "encoder": encoder_identity, "texts_sha256": digest(texts),
                "count": len(texts), "normalize_embeddings": True, "storage_dtype": "float32",
                "implementation": "SentenceTransformer.encode; original model max_seq_length",
                "sentence_transformers": importlib.metadata.version("sentence-transformers")}
    root = args.data_root / "features" / digest(contract)[:20]
    ensure_contract(root / "manifest.json", contract)
    atomic_json(root / "texts.json", texts)
    missing = [i for i in range(0, len(texts), 512) if not (root / f"{i//512:05d}.pt").exists()]
    if missing:
        from sentence_transformers import SentenceTransformer
        progress.log(f"[features] cached={len(texts)-sum(min(512,len(texts)-i) for i in missing)} remaining={sum(min(512,len(texts)-i) for i in missing)}")
        model = SentenceTransformer(str(args.encoder_model), device="cuda:0", local_files_only=True)
        model.eval()
        model.requires_grad_(False)
        dim = model.get_sentence_embedding_dimension()
        cached = len(texts)-sum(min(512, len(texts)-i) for i in missing)
        progress.stage(1, len(texts), initial=cached, unit="text")
        clipped = 0
        for i in missing:
            batch_text = texts[i:i+512]
            clipped += sum(len(model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) > model.max_seq_length
                           for t in batch_text)
            with torch.inference_mode():
                vector = model.encode(batch_text, batch_size=args.embedding_batch_size,
                                      convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            torch_save(root / f"{i//512:05d}.pt", {"text_sha256": digest(batch_text), "values": vector.cpu().float()})
            progress.update(len(batch_text))
        atomic_json(root / "encoder_limits.json", {"max_seq_length": model.max_seq_length,
                    "truncated_texts_this_invocation": clipped, "embedding_dim": dim})
        del model, vector
        gc.collect()
        torch.cuda.empty_cache()
    pieces = []
    for i in range(0, len(texts), 512):
        value = load_torch(root / f"{i//512:05d}.pt")
        if value["text_sha256"] != digest(texts[i:i+512]) or len(value["values"]) != len(texts[i:i+512]):
            raise RuntimeError(f"Embedding shard corrupted: {root}/{i//512:05d}.pt")
        pieces.append(value["values"])
    if not pieces:
        raise RuntimeError("No graph attributes in train/validation; cannot train a graph encoder")
    features = torch.cat(pieces)
    progress.complete(f"frozen attributes={len(texts)}; feature cache={root}")
    return features, {t: i for i, t in enumerate(texts)}, contract


def set_seed(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False


def rng_state():
    import torch
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


@training_attention_backend()
def run_training(args, rows, data_hash, feature_contract, features, lookup, tokenizer, model_id, progress):
    import torch
    from transformers import AutoModelForCausalLM
    from medrag.training.graph_input_modules import GraphPrompt, TextPrompt, answer_losses

    output = args.output_root / args.arm
    contract = {"version": VERSION, "arm": args.arm, "data_sha256": data_hash,
        "model": model_id, "features": feature_contract, "seed": args.seed,
        "architecture": {"frozen_llm": True, "graph_hidden": args.graph_hidden,
            "graph_layers": args.graph_layers, "graph_heads": args.graph_heads,
            "text_soft_tokens": args.text_soft_tokens, "graph_soft_tokens": 1,
            "graph_attention": "native PyTorch edge-aware TransformerConv-style; directed edges, no added reverse edges",
            "reference": "https://github.com/XiaoxinHe/G-Retriever",
            "adaptations": ["Llama-3", "fixed Top-8", "KG-only", "native scatter graph attention"]},
        "training": {k: getattr(args, k) for k in ("epochs", "micro_batch_size", "effective_batch_size",
            "learning_rate", "weight_decay", "max_context", "checkpoint_steps")},
        "loss": "mean per-question teacher-forced gold answer CE; no rationale targets",
        "validation": "validation answer CE only; min CE checkpoint; NO generate or heldout scoring",
        "optimizer": "AdamW; 5% linear warmup then cosine; norm clip=1; LLM BF16, adapters FP32",
        "attention_runtime": "SDPA flash/efficient/math only throughout forward, backward checkpoint recomputation, and validation; cuDNN disabled within run_training",
        "claim_limit": "TEST repartition development; unequal module capacity; not a pure topology causal estimate",
        "environment": {**versions(), "python": platform.python_version()},
        "code": {str(p.relative_to(ROOT)): sha_file(p) for p in
            (Path(__file__), ROOT / "medrag/training/graph_input_modules.py")},
    }
    run_hash = ensure_contract(output / "manifest.json", contract)
    if (output / "COMPLETE.json").exists():
        saved = json.loads((output / "COMPLETE.json").read_text())
        if saved["contract_sha256"] != run_hash:
            raise RuntimeError("Training completion mismatch")
        progress.log(f"[complete cache] {output}; no final evaluation was or will be run")
        return
    # Immutable metadata lives separately from the scientific resume contract.
    if not (output / "provenance.json").exists():
        atomic_json(output / "provenance.json", {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "contract_sha256": run_hash})
    progress.stage(2, 1, unit="model")
    set_seed(args.seed)
    llm = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
    llm.requires_grad_(False)
    llm.eval()
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # HF checkpointing is active only with training=True on the backbone;
    # Llama dropout is zero in this checkpoint. Keep all dropout modules disabled.
    llm.model.train()
    for module in llm.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    emb = llm.get_input_embeddings()
    if args.arm == "text":
        init = tokenizer.encode("Use relevant evidence and answer the question accurately.", add_special_tokens=False)
        init = (init * math.ceil(args.text_soft_tokens / len(init)))[:args.text_soft_tokens]
        adapter = TextPrompt(emb(torch.tensor(init, device="cuda:0")))
    else:
        adapter = GraphPrompt(features.shape[1], llm.config.hidden_size, args.graph_hidden,
                              args.graph_layers, args.graph_heads).to("cuda:0")
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if set(r["sample_id"] for r in train) & set(r["sample_id"] for r in validation):
        raise RuntimeError("Train/validation overlap")
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train) / args.effective_batch_size)
    max_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(.05 * max_steps))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, max_steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    state = {"epoch": 0, "cursor": 0, "step": 0, "best_validation_loss": float("inf"), "history": [],
             "epoch_loss_sum": 0., "epoch_examples": 0}
    last_path = output / "last.pt"
    if last_path.exists():
        saved = load_torch(last_path, trusted=True)  # local, self-generated optimizer/RNG checkpoint only
        if saved["contract_sha256"] != run_hash:
            raise RuntimeError("Checkpoint belongs to a different contract")
        adapter.load_state_dict(saved["adapter"])
        opt.load_state_dict(saved["optimizer"])
        for entry in opt.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.to("cuda:0")
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        restore_rng(saved["rng"])
        del saved
        progress.log(f"[resume] epoch={state['epoch']+1}, durable questions={state['cursor']}/{len(train)}, step={state['step']}")

    def save():
        started = time.monotonic()
        torch_save(last_path, {"contract_sha256": run_hash, "adapter": adapter.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "state": state, "rng": rng_state()})
        atomic_json(output / "history.json", state["history"])
        # Pending finalization is bounded by a measured local checkpoint write,
        # so the overall ETA need not wait until the very end to become known.
        progress.weights[-1] = max(.001, time.monotonic() - started)

    def soft_prompts(batch):
        if args.arm == "text":
            return [adapter() for _ in batch]
        result = []
        for row in batch:
            g = row["graph"]
            ni = torch.tensor([lookup[t] for t in g["node_texts"]], dtype=torch.long)
            ei = torch.tensor([lookup[t] for t in g["edge_texts"]], dtype=torch.long)
            x = features[ni].to("cuda:0")
            edge_attr = features[ei].to("cuda:0")
            index = torch.tensor(g["edge_index"], dtype=torch.long, device="cuda:0")
            result.append(adapter(x, index, edge_attr))
        return result

    def losses(batch):
        soft = soft_prompts(batch)
        return answer_losses(llm, emb, batch, soft, args.arm, tokenizer.pad_token_id)

    trainable = sum(p.numel() for p in adapter.parameters())
    if any(p.requires_grad for p in llm.parameters()):
        raise RuntimeError("LLM must remain frozen")
    atomic_json(output / "parameter_counts.json", {"trainable_adapter": trainable, "trainable_llm": 0,
                "frozen_llm": sum(p.numel() for p in llm.parameters()), "train_questions": len(train),
                "validation_questions": len(validation)})
    progress.complete(f"Llama frozen; adapter parameters={trainable:,}; model={args.model}")
    for completed_epoch in range(state["epoch"]):
        for j, size in ((3+2*completed_epoch, len(train)), (4+2*completed_epoch, len(validation))):
            progress.stage(j, size, initial=size, unit="q")
            progress.complete("durable epoch already completed")
    if not last_path.exists():
        save()
    for epoch in range(state["epoch"], args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        progress.stage(3 + 2*epoch, len(train), initial=state["cursor"], unit="q", kind="train")
        adapter.train()
        llm.model.train()
        for start in range(state["cursor"], len(train), args.effective_batch_size):
            batch = [train[i] for i in order[start:start+args.effective_batch_size]]
            micro = min(args.micro_batch_size, len(batch))
            before_rng = rng_state()
            buffers = {k: v.detach().clone() for k, v in adapter.named_buffers()}
            while True:
                opt.zero_grad(set_to_none=True)
                total_loss = 0.
                try:
                    for i in range(0, len(batch), micro):
                        loss = losses(batch[i:i+micro])
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError("Non-finite training loss")
                        (loss.sum() / len(batch)).backward()
                        total_loss += loss.detach().sum().item()
                        del loss
                    break
                except torch.cuda.OutOfMemoryError:
                    if "loss" in locals():
                        del loss
                    opt.zero_grad(set_to_none=True)
                    for key, buf in adapter.named_buffers():
                        buf.copy_(buffers[key])
                    restore_rng(before_rng)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
                    progress.log(f"[OOM retry] epoch={epoch+1} question={start}; micro={micro}; effective batch unchanged; restore micro={args.micro_batch_size} next step")
            grads = [p.grad for p in adapter.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Missing/non-finite adapter gradients")
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            opt.step()
            scheduler.step()
            state.update(epoch=epoch, cursor=start+len(batch), step=state["step"]+1,
                         epoch_loss_sum=state["epoch_loss_sum"]+total_loss,
                         epoch_examples=state["epoch_examples"]+len(batch))
            if state["step"] % args.checkpoint_steps == 0 or state["cursor"] == len(train):
                save()
            progress.update(len(batch))
        progress.complete(f"epoch={epoch+1}; train CE={state['epoch_loss_sum']/state['epoch_examples']:.6f}; checkpoint={last_path}")
        progress.stage(4 + 2*epoch, len(validation), unit="q", kind="validation")
        adapter.eval()
        llm.model.eval()
        val_sum, per_question = 0., []
        # Validation batches use the same requested micro batch; backprop is disabled.
        with torch.no_grad():
            for i in range(0, len(validation), args.micro_batch_size):
                batch = validation[i:i+args.micro_batch_size]
                micro = len(batch)
                while True:
                    try:
                        values = []
                        for j in range(0, len(batch), micro):
                            values.extend(losses(batch[j:j+micro]).cpu().tolist())
                        break
                    except torch.cuda.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                        if micro == 1:
                            raise
                        micro = max(1, micro // 2)
                        progress.log(f"[validation OOM retry] micro={micro}; next batch restores configured size")
                if any(not math.isfinite(v) for v in values):
                    raise FloatingPointError("Non-finite validation loss")
                val_sum += sum(values)
                per_question.extend({"sample_id": r["sample_id"], "answer_ce": v} for r, v in zip(batch, values, strict=True))
                progress.update(len(batch))
        val_loss = val_sum / len(validation)
        if val_loss < state["best_validation_loss"]:
            state["best_validation_loss"] = val_loss
            torch_save(output / "best.pt", {"contract_sha256": run_hash, "adapter": adapter.state_dict(),
                                            "epoch": epoch+1, "validation_loss": val_loss})
        atomic_jsonl(output / f"validation_loss_epoch{epoch+1}.jsonl", per_question)
        state["history"].append({"epoch": epoch+1, "train_answer_ce": state["epoch_loss_sum"]/state["epoch_examples"],
            "validation_answer_ce": val_loss, "train_n": len(train), "validation_n": len(validation),
            "steps": state["step"], "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        state.update(epoch=epoch+1, cursor=0, epoch_loss_sum=0., epoch_examples=0)
        save()
        progress.complete(f"validation CE={val_loss:.6f}; best={state['best_validation_loss']:.6f}; NO accuracy/generation evaluation")
    progress.stage(len(progress.names)-1, 1)
    atomic_json(output / "COMPLETE.json", {"contract_sha256": run_hash, "epochs": args.epochs,
                "best_validation_answer_ce": state["best_validation_loss"],
                "final_evaluation_executed": False, "pilot_evaluation_questions_used": 0,
                "best_checkpoint": str(output / "best.pt"), "last_checkpoint": str(last_path)})
    progress.update()
    progress.complete(f"training only complete; checkpoints={output}")


def main():
    args = arguments()
    names = ["CPU paired data", "graph attributes", "load Llama"]
    for epoch in range(1, args.epochs+1):
        names += [f"train epoch {epoch}/{args.epochs} mixed", f"validation epoch {epoch}/{args.epochs} mixed"]
    names += ["save completion"]
    progress = Progress(names)
    try:
        _, model_id = check_paths(args)
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        progress.log(f"[plan] {args.arm}: TEST6545 development resplit -> paired train/validation -> "
                     f"{'soft prompt' if args.arm == 'text' else 'frozen SBERT + graph encoder/projector'} -> "
                     f"{args.epochs} training epochs; no final evaluation")
        progress.log("[stages] paired data (resume cache) -> graph attributes (TEXT skips; GRAPH resumes shards) "
                     "-> load frozen Llama -> train/validation per epoch (optimizer checkpoint resume) "
                     "-> save adapter only; uncached durations calibrate per operation")
        progress.log("[ETA] no compatible training throughput yet; train/validation phases calibrate independently. GPU 1 only.")
        data_hash, rows, saved = prepare_data(args, tokenizer, progress)
        progress.log(f"[data] {json.dumps(saved['statistics'])}")
        if args.preflight_only:
            progress.log("[preflight passed] CPU schemas, grouped splits, exact Top-8/KG mapping, BOTH prompt budgets checked; no weights/GPU/download/training/evaluation")
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this process. Run the wrapper on the GPU server; preflight-only remains CPU-safe.")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("This pilot is restricted to physical GPU 1; set CUDA_VISIBLE_DEVICES=1")
        with lock_file(args.output_root / ".gpu_training.lock"):
            ensure_contract(args.output_root / args.arm / "run_plan.json", {
                "version": VERSION, "arguments": {k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items() if k != "preflight_only"},
                "data_contract": data_hash, "model": model_id, "environment": versions(),
                "code_sha256": sha_file(Path(__file__)), "final_evaluation": False})
            features, lookup, feature_contract = prepare_features(args, rows, data_hash, progress)
            run_training(args, rows, data_hash, feature_contract, features, lookup, tokenizer, model_id, progress)
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/{len(names)} item={progress.done}/{progress.total}; "
                     f"{type(exc).__name__}: {exc}. Last durable checkpoint: {args.output_root / args.arm / 'last.pt'}; "
                     "same command resumes compatible completed data shards/optimizer checkpoints; no held-out evaluation.")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

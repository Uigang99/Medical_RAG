#!/usr/bin/env python3
"""Cache four fixed question-only retrieval views for a benchmark split.

One exact inner-product scan of each of four corpora supplies both the balanced and
unrestricted dense pools.  MedCPT CE scores the *union* once per question, so
overlapping documents are not evaluated twice.  Gold answers are never used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from medrag.benchmark import load_benchmark_samples
from medrag.query_embeddings import QueryEmbeddingStore
from medrag.retrieval.inner_product_fourway import derive_candidate_ids, load_state, source_topk_inner_product
from medrag.retrieval.faiss_retriever import FaissSourceIndex
from medrag.retrieval.corpus_filter import load_corpus_filter
from medrag.reranking.medcpt_cross_encoder import MedCPTCrossEncoderReranker
from medrag.training.graph_input_progress import Progress


TEST_DATASETS = (
    "medmcqa", "medqa", "mmlu_anatomy", "mmlu_clinical_knowledge",
    "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics",
    "mmlu_professional_medicine",
)
TRAIN_DATASETS = ("medmcqa", "medqa")
EXPECTED_ROWS = {
    "test": {"medmcqa": 4183, "medqa": 1273, "mmlu_anatomy": 135,
             "mmlu_clinical_knowledge": 265, "mmlu_college_biology": 144,
             "mmlu_college_medicine": 173, "mmlu_medical_genetics": 100,
             "mmlu_professional_medicine": 272},
    "train": {"medmcqa": 182822, "medqa": 10178},
}
SOURCES = ("pubmed", "pmc", "cpg", "textbooks")
ARMS = ("balanced_dense", "global_dense", "balanced_ce", "global_ce")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names; defaults to all eight TEST datasets or MedMCQA+MedQA TRAIN")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=PROJECT / "datasets" / "benchmark" /
                        "mcq" / "unified")
    parser.add_argument("--query-cache-root", type=Path, default=PROJECT / "databases" /
                        "query_embeddings" / "medcpt_query_encoder_question_only" / "mcq" / "unified")
    parser.add_argument("--vector-db-root", type=Path, default=PROJECT / "databases" / "vector_db" /
                        "RAG_Square")
    parser.add_argument("--corpus-filter", type=Path, default=None,
                        help="Frozen eligibility masks; restrict before exact Top-32 search")
    parser.add_argument("--ce-model", type=Path, default=PROJECT.parent / "models" / "MedCPT-Cross-Encoder")
    parser.add_argument("--vector-chunk-size", type=int, default=100_000)
    parser.add_argument("--query-batch-size", type=int, default=2_048)
    parser.add_argument("--ce-batch-size", type=int, default=128)
    parser.add_argument("--ce-question-batch-size", type=int, default=16)
    parser.add_argument("--ce-max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0", help="Visible device; wrapper exposes physical GPU 1 as cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan; no GPU or output writes")
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = TEST_DATASETS if args.split == "test" else TRAIN_DATASETS
    if len(set(args.datasets)) != len(args.datasets) or any(
            dataset not in EXPECTED_ROWS[args.split] for dataset in args.datasets):
        parser.error(f"Invalid or repeated datasets for {args.split}: {args.datasets}")
    if args.output_root is None:
        count = sum(EXPECTED_ROWS[args.split][dataset] for dataset in args.datasets)
        args.output_root = (PROJECT / "databases" / "run_cache" /
                            f"question_only_ip_fourway_{args.split}{count}_v1")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def preflight(args: argparse.Namespace) -> tuple[list, np.ndarray, dict, str]:
    """Verify the requested cohort, query text, embedding cache, and DB contract."""
    samples = []
    vector_batches = []
    inputs = []
    query_digest = hashlib.sha256()
    for dataset in args.datasets:
        benchmark = args.benchmark_root / dataset / f"{args.split}.jsonl"
        cache_dir = args.query_cache_root / dataset / args.split
        manifest_path = cache_dir / "manifest.json"
        metadata_path = cache_dir / "metadata.jsonl"
        if not benchmark.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing question-only input for {dataset}: {benchmark}, {cache_dir}")
        cohort = load_benchmark_samples(benchmark, task="mcq", collection="unified",
                                        dataset=dataset, split=args.split)
        store = QueryEmbeddingStore(cache_dir)
        manifest = store.manifest
        if manifest.get("mcq_query_mode") != "question_only" or manifest.get("normalize") is not False:
            raise RuntimeError(f"Not the expected unnormalized question-only MedCPT cache: {manifest_path}")
        if manifest.get("model", {}).get("name") != "MedCPT-Query-Encoder":
            raise RuntimeError(f"Wrong query model: {manifest_path}")
        if (len(cohort) != EXPECTED_ROWS[args.split][dataset] or
                len(cohort) != int(manifest["rows"]) or store.dimension != 768):
            raise RuntimeError(f"Question count/dimension mismatch: {dataset}")
        with metadata_path.open(encoding="utf-8") as handle:
            for sample, line in zip(cohort, handle, strict=True):
                metadata = json.loads(line)
                # This older cache stores the embedding-batch start in
                # metadata.row_idx (0, 2048, ...), not each row's position.
                # File order, ID, and exact query text are the valid alignment.
                if (metadata.get("id") != sample.id or
                    metadata.get("query_text") != sample.question):
                    raise RuntimeError(f"Query text/ID mismatch for {dataset}:{sample.row_idx}")
                query_digest.update((dataset + "\0" + sample.id + "\0" + sample.question + "\n").encode())
        samples.extend(cohort)
        vector_batches.append(store.get_batch([sample.row_idx for sample in cohort]))
        inputs.append({"dataset": dataset, "rows": len(cohort), "benchmark_sha256": sha256_file(benchmark),
                       "query_manifest_sha256": sha256_file(manifest_path),
                       "query_embeddings": signature(cache_dir / "embeddings.npy"),
                       "query_metadata": signature(metadata_path)})
    expected = sum(EXPECTED_ROWS[args.split][dataset] for dataset in args.datasets)
    if len(samples) != expected or len({sample.id for sample in samples}) != expected:
        raise RuntimeError(f"Expected exactly {expected} unique {args.split} questions; got {len(samples)}")

    source_contract = []
    for source in SOURCES:
        manifest_path = args.vector_db_root / source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("source") != source or manifest.get("index", {}).get("dimension") != 768 or
            manifest.get("index", {}).get("metric") != "inner_product" or
            manifest.get("model", {}).get("name") != "MedCPT-Article-Encoder" or
            manifest.get("model", {}).get("normalize") is not False):
            raise RuntimeError(f"Unexpected corpus vector contract: {manifest_path}")
        index_files = [args.vector_db_root / source / shard["path"]
                       for shard in manifest["index"]["shards"]]
        if not all(path.is_file() for path in index_files):
            raise FileNotFoundError(f"Missing FAISS shard in {args.vector_db_root / source}")
        metadata_files = [args.vector_db_root / source / shard["path"]
                          for shard in manifest["metadata"]["shards"]]
        offset_files = [path.with_suffix(".offsets.npy") for path in metadata_files]
        if len(metadata_files) != len(index_files) or not all(
                path.is_file() for path in [*metadata_files, *offset_files]):
            raise FileNotFoundError(f"Missing metadata or random-access offsets: {args.vector_db_root / source}")
        source_contract.append({"source": source, "rows": int(manifest["rows"]),
                                "manifest_sha256": sha256_file(manifest_path),
                                "shards": [signature(path) for path in index_files],
                                "metadata": [signature(path) for path in metadata_files],
                                "metadata_offsets": [signature(path) for path in offset_files]})
    if not args.ce_model.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Missing MedCPT CE model: {args.ce_model}")
    ce_weights = sorted([*args.ce_model.glob("*.safetensors"), *args.ce_model.glob("*.bin")])
    if not ce_weights:
        raise FileNotFoundError(f"Missing MedCPT CE weights: {args.ce_model}")
    if any(value <= 0 for value in (args.vector_chunk_size, args.query_batch_size,
                                    args.ce_batch_size, args.ce_question_batch_size, args.ce_max_length)):
        raise ValueError("Chunk and batch sizes must be positive")
    vectors = np.ascontiguousarray(np.concatenate(vector_batches), dtype=np.float32)
    if vectors.shape != (expected, 768) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Invalid {args.split} question embedding matrix")
    contract = {
        "experiment": f"question_only_ip_fourway_{args.split}{expected}_v1",
        "split": args.split, "datasets": list(args.datasets),
        "question_input": "benchmark question only; no options, rationale, or gold",
        "retrieval_metric": "raw float32 inner product; stored unnormalized FAISS vectors are reconstructed",
        "sources": list(SOURCES), "source_contract": source_contract,
        "dataset_contract": inputs, "question_id_text_sha256": query_digest.hexdigest(),
        "balanced": "top 8 from each source, then inner-product merge of 32",
        "global": "global inner-product top 32 from per-source top 32",
        "ce": {"model_path": str(args.ce_model.resolve()),
               "config_sha256": sha256_file(args.ce_model / "config.json"),
               "weight_files": [signature(path) for path in ce_weights],
               "question_only": True, "max_length": args.ce_max_length,
               "batch_size": args.ce_batch_size, "question_batch_size": args.ce_question_batch_size},
        "vector_chunk_size": args.vector_chunk_size, "query_batch_size": args.query_batch_size,
        "code_sha256": {str(path.relative_to(PROJECT)): sha256_file(path) for path in
                        (Path(__file__), PROJECT / "medrag" / "retrieval" / "inner_product_fourway.py")},
    }
    if args.corpus_filter is not None:
        args.eligibility_masks, filter_contract = load_corpus_filter(args.corpus_filter, source_contract)
        contract["corpus_filter"] = filter_contract
        contract["code_sha256"]["medrag/retrieval/corpus_filter.py"] = sha256_file(
            PROJECT / "medrag/retrieval/corpus_filter.py")
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract["contract_sha256"] = contract_hash
    contract["output_root"] = str(args.output_root.resolve())
    try:
        contract["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        contract["git_commit"] = None
    return samples, vectors, contract, contract_hash


def checked_manifest(output_root: Path, contract: dict) -> None:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"Output contract mismatch; use a new --output-root: {output_root}")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError(f"Nonempty output without manifest: {output_root}")
        atomic_json(manifest_path, contract)


def source_checkpoint(output_root: Path, source: str) -> Path:
    return output_root / "inner_product_checkpoints" / f"{source}.npz"


def completed_vectors(source_index: FaissSourceIndex, checkpoint_path: Path,
                      contract_hash: str, questions: int) -> int:
    shards = source_index.manifest["index"]["shards"]
    cursor, _, _ = load_state(checkpoint_path, contract=contract_hash,
                              shape=(questions, 32), shard_count=len(shards))
    return sum(int(shard["rows"]) for shard in shards[:cursor])


def _doc_record(source_indexes: dict[str, FaissSourceIndex], hit: tuple,
                dense_rank: int, document_cache: dict[tuple[str, int], object]) -> dict:
    source, local_id, score, source_rank = hit
    key = (source, local_id)
    if key not in document_cache:
        document_cache[key] = source_indexes[source].get_document(local_id, score)
    record = document_cache[key].to_dict(include_text=True)
    record.update(inner_product_score=score, retrieval_score=score, retrieval_rank=dense_rank,
                  source_rank=source_rank, dense_rank=dense_rank, rank=dense_rank,
                  ce_score=None, ce_rank=None)
    return record


def _ce_rank(dense: list[dict], score_map: dict[tuple[str, int], float]) -> list[dict]:
    ranked = []
    for record in dense:
        score = score_map[(record["source"], record["local_id"])]
        ranked.append(dict(record, ce_score=score, rerank_score=score))
    ranked.sort(key=lambda doc: (-doc["ce_score"], doc["dense_rank"]))
    for rank, record in enumerate(ranked, start=1):
        record["ce_rank"] = rank
        record["rerank_rank"] = rank
        record["rank"] = rank
    return ranked


def valid_part(path: Path, samples: list, start: int, end: int, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) != end - start:
        raise RuntimeError(f"Incomplete durable part: {path}")
    for sample, line in zip(samples[start:end], lines, strict=True):
        row = json.loads(line)
        if row.get("sample_id") != sample.id or row.get("contract_sha256") != contract_hash:
            raise RuntimeError(f"Part contract/ID mismatch: {path}")
        if any(len(row.get(arm, [])) != 32 for arm in ARMS):
            raise RuntimeError(f"Part does not contain four Top-32 lists: {path}")
    return True


def rerank_parts(args: argparse.Namespace, samples: list,
                 per_source: dict[str, tuple[np.ndarray, np.ndarray]],
                 source_indexes: dict[str, FaissSourceIndex], contract_hash: str,
                 progress: Progress, *, stage_index: int = 5,
                 balanced_quotas: dict[str, int] | None = None) -> None:
    count = (len(samples) + args.ce_question_batch_size - 1) // args.ce_question_batch_size
    part_dir = args.output_root / "parts"
    ranges = [(part * args.ce_question_batch_size,
               min(len(samples), (part + 1) * args.ce_question_batch_size))
              for part in range(count)]
    complete = [valid_part(part_dir / f"part_{part:05d}.jsonl", samples, start, end, contract_hash)
                for part, (start, end) in enumerate(ranges)]
    progress.stage(stage_index, count, initial=sum(complete), unit="batch", kind="ce")
    if all(complete):
        progress.complete(f"cached {count} CE batches; {part_dir}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("MedCPT CE requires GPU 1; run via the supplied wrapper on the H200 host")
    reranker = MedCPTCrossEncoderReranker(args.ce_model, batch_size=args.ce_batch_size,
                                          max_length=args.ce_max_length, device=args.device,
                                          attn_implementation="sdpa")
    try:
        for part, (start, end) in enumerate(ranges):
            if complete[part]:
                continue
            dense_rows = []
            union_lists = []
            for row_index in range(start, end):
                balanced, global_hits = derive_candidate_ids(per_source, row_index,
                                                             balanced_quotas=balanced_quotas)
                cache: dict[tuple[str, int], object] = {}
                balanced_docs = [_doc_record(source_indexes, hit, rank, cache)
                                 for rank, hit in enumerate(balanced, start=1)]
                global_docs = [_doc_record(source_indexes, hit, rank, cache)
                               for rank, hit in enumerate(global_hits, start=1)]
                dense_rows.append((balanced_docs, global_docs))
                union_lists.append(list(cache.values()))
            queries = [sample.question for sample in samples[start:end]]
            ce_lists = reranker.rerank_batch(queries, union_lists, top_k=64)
            rows = []
            for sample, (balanced_docs, global_docs), ce_docs in zip(
                    samples[start:end], dense_rows, ce_lists, strict=True):
                score_map = {(doc.source, doc.local_id): float(doc.rerank_score) for doc in ce_docs}
                if any((doc["source"], doc["local_id"]) not in score_map
                       for doc in balanced_docs + global_docs):
                    raise RuntimeError(f"CE score missing for {sample.id}")
                rows.append({"contract_sha256": contract_hash, "dataset": sample.dataset,
                             "sample_id": sample.id, "row_idx": sample.row_idx,
                             "query_text": sample.question,
                             "balanced_dense": balanced_docs, "global_dense": global_docs,
                             "balanced_ce": _ce_rank(balanced_docs, score_map),
                             "global_ce": _ce_rank(global_docs, score_map)})
            atomic_jsonl(part_dir / f"part_{part:05d}.jsonl", rows)
            progress.update(1)
        progress.complete(f"CE scored each question's unique union once; {part_dir}")
    finally:
        reranker.close()


def merge_outputs(args: argparse.Namespace, samples: list, contract_hash: str,
                  progress: Progress, *, stage_index: int = 6) -> None:
    part_dir = args.output_root / "parts"
    count = (len(samples) + args.ce_question_batch_size - 1) // args.ce_question_batch_size
    progress.stage(stage_index, count, unit="batch", kind="merge")
    output_paths = {arm: args.output_root / f"{arm}.jsonl" for arm in ARMS}
    temporary = {arm: path.with_suffix(".jsonl.tmp") for arm, path in output_paths.items()}
    handles = {arm: path.open("w", encoding="utf-8") for arm, path in temporary.items()}
    source_counts = {arm: Counter() for arm in ARMS}
    try:
        for part in range(count):
            start = part * args.ce_question_batch_size
            end = min(len(samples), start + args.ce_question_batch_size)
            path = part_dir / f"part_{part:05d}.jsonl"
            if not valid_part(path, samples, start, end, contract_hash):
                raise RuntimeError(f"Missing CE part: {path}")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    common = {key: row[key] for key in ("dataset", "sample_id", "row_idx", "query_text")}
                    for arm in ARMS:
                        docs = row[arm]
                        source_counts[arm].update(doc["source"] for doc in docs)
                        handles[arm].write(json.dumps({**common, "documents": docs}, ensure_ascii=False) + "\n")
            progress.update(1)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    for arm, path in output_paths.items():
        os.replace(temporary[arm], path)
    summary = {"questions": len(samples), "split": args.split,
               "datasets": {dataset: EXPECTED_ROWS[args.split][dataset] for dataset in args.datasets},
               "documents_per_question_per_arm": 32,
               "metric": "inner_product", "arms": {arm: {"path": str(output_paths[arm]),
               "source_counts": dict(source_counts[arm])} for arm in ARMS}}
    summary["contract_sha256"] = contract_hash
    if args.corpus_filter is not None:
        summary["corpus_filter"] = str(args.corpus_filter.resolve())
    atomic_json(args.output_root / "summary.json", summary)
    progress.complete(f"four {len(samples)}-row ranked JSONL files and summary: {args.output_root}")


def run(args: argparse.Namespace) -> None:
    samples, vectors, contract, contract_hash = preflight(args)
    # The completed TEST run used ~2.2 GB for 6,545 rows including durable parts.
    # Reserve a conservative margin before starting a fresh large run.
    if not args.dry_run and not args.output_root.exists():
        required = int(len(samples) * 350_000 * 1.5)
        available = shutil.disk_usage(args.output_root.parent).free
        if available < required:
            raise RuntimeError(f"Insufficient free disk: {available / 1e9:.1f} GB; "
                               f"need ~{required / 1e9:.1f} GB for a fresh run")
    source_rows = {item["source"]: item["rows"] for item in contract["source_contract"]}
    total = sum(source_rows.values())
    plan = (f"[plan] {args.split.upper()}{len(samples)} question-only MedCPT -> four corpus inner-product scans "
            "-> balanced/global 32 -> deduplicated MedCPT CE -> four ranked JSONL; "
            f"vectors={total:,}, GPU 1 only, no labels, no generation")
    print(plan, flush=True)
    if args.corpus_filter is not None:
        print(f"[filter] {contract['corpus_filter']['eligible_rows']:,} eligible vectors; "
              "original IDs/text preserved; filter before Top-32, never post-filter old hits", flush=True)
    # Approximate stage durations from the completed, same-GPU TEST-6545 run:
    # search 278 s, CE 312 s, merge 13 s. Recalibrate from fresh work.
    scale = len(samples) / 6545
    search_seconds, ce_seconds, merge_seconds = 278 * scale, 312 * scale, 13 * scale
    print("[stages] contract; pubmed; pmc; cpg; textbooks; CE rerank; merge. "
          f"Previous-run scaled ETA ~{(search_seconds + ce_seconds + merge_seconds) / 3600:.1f} h "
          "(same H200 GPU 1 and batch sizes); recalibrates during execution.", flush=True)
    if args.dry_run:
        print(f"[dry-run OK] N={len(samples)} datasets={','.join(args.datasets)}; "
              f"query mode=question_only; output={args.output_root}; "
              "no GPU/model/index load and no output writes", flush=True)
        return
    checked_manifest(args.output_root, contract)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Execute on H200 GPU 1 with the supplied wrapper")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This 125M-vector retrieval requires CUDA")
    progress = Progress(["input contract", *[f"inner product {source}" for source in SOURCES],
                         "MedCPT CE union", "merge four outputs"])
    progress.weights[0] = 1.0
    for stage, source in enumerate(SOURCES, start=1):
        progress.weights[stage] = search_seconds * source_rows[source] / total
    progress.weights[5] = ce_seconds
    progress.weights[6] = merge_seconds
    current = "contract"
    indexes = {}
    try:
        progress.stage(0, 1, unit="check")
        progress.complete(f"manifest={args.output_root / 'manifest.json'}; N={len(samples)}")
        per_source = {}
        for stage, source in enumerate(SOURCES, start=1):
            current = f"inner product {source}"
            index = FaissSourceIndex(args.vector_db_root / source, cache_index=False,
                                     mmap_index=True, metadata_row_cache_size=50_000)
            indexes[source] = index
            checkpoint = source_checkpoint(args.output_root, source)
            prior = completed_vectors(index, checkpoint, contract_hash, len(samples))
            progress.stage(stage, index.rows, initial=prior, unit="vector", kind="inner_product")
            stage_start = time.monotonic()

            def on_vectors(amount: int) -> None:
                progress.update(amount)
                fresh = progress.done - progress.initial
                if fresh > 0:
                    seconds_per_vector = (time.monotonic() - stage_start) / fresh
                    for future_stage, future_source in enumerate(SOURCES, start=1):
                        if future_stage > stage:
                            progress.weights[future_stage] = max(
                                1., seconds_per_vector * source_rows[future_source])

            scores, ids = source_topk_inner_product(
                index, vectors, k=32, vector_chunk_size=args.vector_chunk_size,
                query_batch_size=args.query_batch_size, device=device,
                checkpoint_path=checkpoint, contract=contract_hash, on_vectors=on_vectors,
                on_checkpoint=lambda done, count: progress.log(
                    f"[checkpoint] {source} shard {done}/{count}: {checkpoint}"),
                eligibility=getattr(args, "eligibility_masks", {}).get(source),
            )
            if args.corpus_filter is not None and not args.eligibility_masks[source][ids].all():
                raise RuntimeError(f"Ineligible search result in {source}")
            per_source[source] = scores, ids
            progress.complete(f"{index.rows:,} vectors; 32 hits/question; checkpoint={checkpoint}")
        torch.cuda.empty_cache()
        current = "MedCPT CE"
        rerank_parts(args, samples, per_source, indexes, contract_hash, progress)
        current = "merge"
        merge_outputs(args, samples, contract_hash, progress)
        progress.log(f"[complete] N={len(samples)}; four Top-32 sets: {args.output_root}")
    except Exception as exc:
        progress.log(f"[FAILED] stage={progress.index + 1}/7 {current}; "
                     f"completed={progress.done}/{progress.total} remaining={progress.total-progress.done}; "
                     f"durable={args.output_root}; same command resumes completed shards/CE parts; "
                     f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for index in indexes.values():
            index.close()
        progress.close()


if __name__ == "__main__":
    run(arguments())

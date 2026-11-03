#!/usr/bin/env python3
"""Evaluate the PCED amateur prompt alone on the same five LOFT-RAG test sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.training.graph_input_progress import Progress
from scripts import run_loft_pced_paper as paper

DATASETS = paper.DATASETS
MAX_NEW_TOKENS = 32


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pced-root", type=Path,
                   default=ROOT / "results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "results/loft_pced_no_rag_v1_full500")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    a.pced_root, a.output_root = a.pced_root.resolve(), a.output_root.resolve()
    if a.pced_root == a.output_root:
        p.error("No-RAG results must not overwrite PCED results")
    return a


def no_rag_ids(tokenizer, question):
    """Concatenate separately tokenized PCED amateur prefix and question suffix."""
    prefixes, suffix = paper.prompt_inputs(tokenizer, question, [])
    return (tokenizer(prefixes[0], add_special_tokens=False)["input_ids"] +
            tokenizer(suffix, add_special_tokens=False)["input_ids"])


def make_batch(rows, pad_id, device):
    import torch
    width = max(len(row[2]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, (_, _, tokens) in enumerate(rows):
        ids[index, -len(tokens):] = torch.tensor(tokens, dtype=torch.long)
        mask[index, -len(tokens):] = 1
    return ids.to(device), mask.to(device)


def preflight(a):
    rows, references = [], {}
    for dataset in DATASETS:
        source = paper.arguments(["--phase", "preflight", "--dataset", dataset,
                                  "--output-root", str(a.pced_root)])
        _, query_file, _, queries = paper.load_inputs(source)
        reference = a.pced_root / dataset / "manifest.json"
        summary = a.pced_root / dataset / "summary.json"
        if not reference.is_file() or not summary.is_file():
            raise FileNotFoundError(f"Completed PCED reference missing: {dataset}")
        ref = json.loads(reference.read_text())
        score = json.loads(summary.read_text())
        if (ref.get("queries_sha256") != paper.sha(query_file) or
                ref.get("llama") != str(source.llama) or ref.get("max_new_tokens") != MAX_NEW_TOKENS or
                ref.get("attention_impl") != "sdpa" or ref.get("questions") != 100 or
                score.get("n") != 100 or score.get("missing") != 0):
            raise RuntimeError(f"PCED cohort/model/decode contract mismatch: {dataset}")
        references[dataset] = {"manifest_sha256": paper.sha(reference),
                               "queries_sha256": paper.sha(query_file)}
        rows.extend((dataset, row) for row in queries)
    model = Path(json.loads((a.pced_root / DATASETS[0] / "manifest.json").read_text())["llama"])
    for name in ("config.json", "tokenizer.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    expected = {"version": "loft_pced_amateur_only_v1", "n": 500,
                "datasets": list(DATASETS), "reference": references,
                "llama": str(model), "gpu": 1, "batch_size": a.batch_size,
                "prompt": "exact PCED amateur stream: original system and question templates, no document",
                "prompt_source_sha256": paper.sha(Path(paper.__file__)),
                "decode": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS,
                           "backend": "sdpa", "seed": 42},
                "metric": "paper Subspan EM", "script_sha256": paper.sha(Path(__file__))}
    manifest = a.output_root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != expected:
        raise RuntimeError(f"Incompatible No-RAG resume contract: {manifest}")
    return rows, model, expected, manifest


def main(argv=None):
    a = arguments(argv)
    progress = Progress(["reference contract", "tokenize amateur prompts", "load frozen Llama",
                         "greedy No-RAG 500q", "Subspan EM summary"])
    progress.weights[:] = [1., 2., 3., None, 1.]
    try:
        progress.stage(0, 1)
        rows, model_path, expected, manifest = preflight(a)
        cached = sum(paper.result_file(a.output_root / dataset / "predictions", str(q["qid"])).is_file()
                     for dataset, q in rows)
        progress.complete(f"same 5x100 test questions, same Llama; cached={cached}/500")
        progress.log(f"[plan] GPU1 only; PCED amateur prompt without documents; greedy 32 tokens; batch={a.batch_size}; cached={cached}, remaining={500-cached}; no retrieval or gold in generation")
        if a.dry_run:
            progress.log(f"[dry-run OK] CPU-only; no model loaded or output written; root={a.output_root}")
            return
        if not manifest.exists():
            paper.atomic_json(manifest, expected)
        progress.stage(1, 500)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                  padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        pending = []
        for dataset, q in rows:
            path = paper.result_file(a.output_root / dataset / "predictions", str(q["qid"]))
            if not path.is_file():
                pending.append((dataset, q, no_rag_ids(tokenizer, q["query_text"])))
            progress.update()
        progress.complete(f"tokenized={len(pending)} pending; output={a.output_root}")
        progress.stage(2, 1)
        if pending:
            import torch
            from transformers import AutoModelForCausalLM
            paper.gpu_gate(40)
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(42)
            torch.manual_seed(42)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        progress.complete(f"frozen Llama; pending={len(pending)}")
        progress.stage(3, 500, initial=cached, unit="q")
        for start in range(0, len(pending), a.batch_size):
            batch = pending[start:start + a.batch_size]
            input_ids, attention_mask = make_batch(batch, tokenizer.pad_token_id, model.device)
            began = time.monotonic()
            with torch.inference_mode():
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                        pad_token_id=tokenizer.pad_token_id, use_cache=True)
            elapsed = time.monotonic() - began
            for index, (dataset, q, tokens) in enumerate(batch):
                generated = output[index, input_ids.shape[1]:].tolist()
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                paper.atomic_json(paper.result_file(a.output_root / dataset / "predictions", str(q["qid"])), {
                    "dataset": dataset, "qid": str(q["qid"]), "prediction": prediction,
                    "generated_token_ids": generated, "answers": q["answers"],
                    "subspan_em": paper.subspan_em(prediction, q["answers"]),
                    "input_tokens": len(tokens), "batch_seconds_per_question": elapsed / len(batch)})
                progress.update()
            del input_ids, attention_mask, output
        progress.complete(f"predictions=500/500; {a.output_root}")
        progress.stage(4, 1)
        scores = {}
        for dataset in DATASETS:
            records = [json.loads(paper.result_file(a.output_root / dataset / "predictions",
                                                   str(q["qid"])).read_text())
                       for name, q in rows if name == dataset]
            correct = sum(record["subspan_em"] for record in records)
            scores[dataset] = {"n": 100, "correct": correct, "subspan_em": correct / 100,
                               "missing": 0}
        correct = sum(value["correct"] for value in scores.values())
        paper.atomic_json(a.output_root / "summary.json", {
            "n": 500, "correct": correct, "micro_subspan_em": correct / 500,
            "macro5_subspan_em": sum(value["subspan_em"] for value in scores.values()) / 5,
            "datasets": scores, "manifest": str(manifest)})
        progress.complete(f"{a.output_root/'summary.json'}")
        progress.log(f"[result] No-RAG {correct}/500; output={a.output_root}")
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/5; {exc}; durable={a.output_root}; rerun same command to resume compatible predictions")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

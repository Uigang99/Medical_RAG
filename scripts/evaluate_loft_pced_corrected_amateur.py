#!/usr/bin/env python3
"""Matched LOFT-RAG PCED/No-RAG comparison with a document-free amateur prompt.

The completed paper-reproduction ranking and expert-score caches are read-only.
Expert prompts and PCED decoding stay unchanged; only the amateur prompt is
rewritten to make sense without documents.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.training.graph_input_progress import Progress
from scripts import run_loft_pced_paper as paper
from scripts.evaluate_loft_pced_no_rag import make_batch

AMATEUR_SYSTEM = "Answer the user's query accurately and concisely using your own knowledge."
AMATEUR_QUESTION = ("Answer the following query using your own knowledge. "
                    "Write a concise answer.\nquery: {question}")
MAX_NEW_TOKENS = 32


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("pced", "no-rag"), required=True)
    p.add_argument("--reference-root", type=Path,
                   default=ROOT / "results/loft_pced_paper_v1_full500_parallel2")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--batch-size", type=int, default=16, help="No-RAG only")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    a.reference_root = a.reference_root.resolve()
    a.output_root = (a.output_root or ROOT / "results" /
                     f"loft_pced_amateur_aware_{a.arm.replace('-', '_')}_v1_500").resolve()
    if a.output_root == a.reference_root:
        p.error("New prompt results must not overwrite the paper-reproduction output")
    return a


def split_prompt(tokenizer, system, context):
    """Keep the official PCED document prefix/suffix split intact."""
    separator = "\n" + "#" * (len(context) + 1)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": context + separator}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
    prefix, suffix = rendered.split(separator)
    return prefix, suffix


def prompt_inputs(tokenizer, question, texts):
    amateur_prefix, amateur_tail = split_prompt(tokenizer, AMATEUR_SYSTEM, "")
    expert_prefixes = []
    expert_tail = None
    for text in texts:
        prefix, tail = split_prompt(tokenizer, paper.SYSTEM, text)
        expert_prefixes.append(prefix)
        if expert_tail is None:
            expert_tail = tail
        elif tail != expert_tail:
            raise RuntimeError("Expert question suffix differs across documents")
    if expert_tail is None:
        _, expert_tail = split_prompt(tokenizer, paper.SYSTEM, "")
    amateur_question = AMATEUR_QUESTION.format(question=question) + amateur_tail + paper.ANSWER_PREFIX
    expert_question = paper.QUESTION.format(question=question) + expert_tail + paper.ANSWER_PREFIX
    return [amateur_prefix, *expert_prefixes], [amateur_question, *([expert_question] * len(texts))]


def no_rag_ids(tokenizer, question):
    prefixes, questions = prompt_inputs(tokenizer, question, [])
    return (tokenizer(prefixes[0], add_special_tokens=False)["input_ids"] +
            tokenizer(questions[0], add_special_tokens=False)["input_ids"])


def preflight(a):
    official = ROOT / "cache/pced_official"
    source_file = official / "src/methods/pced.py"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    revision = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != paper.OFFICIAL_SHA:
        raise RuntimeError(f"Official PCED decoder revision changed: {revision}")
    datasets, references = {}, {}
    for dataset in paper.DATASETS:
        source = paper.arguments(["--phase", "preflight", "--dataset", dataset,
                                  "--output-root", str(a.reference_root)])
        _, query_file, corpus, queries = paper.load_inputs(source)
        reference = a.reference_root / dataset / "manifest.json"
        summary = a.reference_root / dataset / "summary.json"
        if not reference.is_file() or not summary.is_file():
            raise FileNotFoundError(f"Completed paper PCED reference missing: {dataset}")
        ref = json.loads(reference.read_text())
        score = json.loads(summary.read_text())
        if (ref.get("queries_sha256") != paper.sha(query_file) or
                ref.get("llama") != str(source.llama) or ref.get("questions") != 100 or
                ref.get("top_k") != 90 or ref.get("max_new_tokens") != MAX_NEW_TOKENS or
                ref.get("candidate_mode") != "all" or ref.get("score_policy") != "paper" or
                ref.get("attention_impl") != "sdpa" or score.get("n") != 100 or
                score.get("missing") != 0):
            raise RuntimeError(f"Reference PCED contract mismatch: {dataset}")
        references[dataset] = {"manifest_sha256": paper.sha(reference),
                               "queries_sha256": paper.sha(query_file)}
        datasets[dataset] = (queries, {str(doc["pid"]): doc for doc in corpus})
        if a.arm == "pced":
            for q in queries:
                qid = str(q["qid"])
                for kind in ("rankings", "expert_scores"):
                    if not paper.result_file(a.reference_root / dataset / kind, qid).is_file():
                        raise FileNotFoundError(f"Missing fixed PCED {kind}: {dataset}/{qid}")
    model = Path(json.loads((a.reference_root / paper.DATASETS[0] / "manifest.json").read_text())["llama"])
    for filename in ("config.json", "tokenizer.json"):
        if not (model / filename).is_file():
            raise FileNotFoundError(model / filename)
    expected = {"version": "loft_pced_amateur_aware_v1", "arm": a.arm,
                "datasets": list(paper.DATASETS), "n": 500, "reference": references,
                "reference_root": str(a.reference_root), "output_root": str(a.output_root),
                "llama": str(model), "gpu": 1,
                "expert_system": paper.SYSTEM, "expert_question": paper.QUESTION,
                "amateur_system": AMATEUR_SYSTEM, "amateur_question": AMATEUR_QUESTION,
                "answer_prefix": paper.ANSWER_PREFIX,
                "decode": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS,
                           "backend": "sdpa", "seed": 42},
                "batch_size": a.batch_size if a.arm == "no-rag" else None,
                "retrieval": ("fixed paper Top-90, BGE-M3 dense + CE prior" if a.arm == "pced" else "none"),
                "metric": "paper Subspan EM", "script_sha256": paper.sha(Path(__file__)),
                "paper_core_sha256": paper.sha(Path(paper.__file__)),
                "official_pced_commit": revision,
                "git_commit": subprocess.check_output(["git", "-C", str(ROOT),
                                                        "rev-parse", "HEAD"], text=True).strip(),
                "environment": {"python": sys.version.split()[0],
                                "torch": importlib.metadata.version("torch"),
                                "transformers": importlib.metadata.version("transformers")}}
    manifest = a.output_root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != expected:
        raise RuntimeError(f"Incompatible corrected-amateur resume contract: {manifest}")
    return datasets, model, expected, manifest


def run_pced(a, datasets, model, tokenizer, progress):
    import torch
    sys.path.insert(0, str(ROOT / "cache/pced_official"))
    from src.methods.pced import prefill, decode_from_cache
    total = sum(len(queries) for queries, _ in datasets.values())
    cached = sum(paper.result_file(a.output_root / dataset / "predictions", str(q["qid"])).is_file()
                 for dataset, (queries, _) in datasets.items() for q in queries)
    progress.stage(2, total, initial=cached, unit="q")
    for dataset in paper.DATASETS:
        queries, by_pid = datasets[dataset]
        for q in queries:
            qid = str(q["qid"])
            output_path = paper.result_file(a.output_root / dataset / "predictions", qid)
            if output_path.is_file():
                continue
            ranking = json.loads(paper.result_file(a.reference_root / dataset / "rankings", qid).read_text())
            scores = json.loads(paper.result_file(a.reference_root / dataset / "expert_scores", qid).read_text())
            selected = ranking["top90"]
            if len(selected) != 90 or scores["pids"] != [item["pid"] for item in selected]:
                raise RuntimeError(f"Top-90 ordering mismatch: {dataset}/{qid}")
            texts = [paper.format_doc(by_pid[item["pid"]]) for item in selected]
            prefixes, questions = prompt_inputs(tokenizer, q["query_text"], texts)
            if len(prefixes) != 91 or len(questions) != 91:
                raise RuntimeError(f"Expected amateur + 90 experts: {dataset}/{qid}")
            tokenized_prefix = tokenizer(prefixes, return_tensors="pt", add_special_tokens=False,
                                         padding=True, truncation=True,
                                         max_length=model.config.max_position_embeddings - MAX_NEW_TOKENS).to(model.device)
            tokenized_question = tokenizer(questions, return_tensors="pt", add_special_tokens=False,
                                           padding=True).to(model.device)
            if not bool(tokenized_question["attention_mask"][1:].all().item()):
                raise RuntimeError(f"Expert question unexpectedly padded: {dataset}/{qid}")
            prior = paper.prior_tensor(scores, "paper", model.device, model.dtype)
            started = time.monotonic()
            with torch.inference_mode():
                cache = prefill(model, tokenized_prefix)
                ids, winners = decode_from_cache(model, tokenizer, cache, tokenized_prefix,
                    tokenized_question, max_new_tokens=MAX_NEW_TOKENS, beta=None,
                    retrieval_scores=prior, gamma=2.5, normalize=False,
                    return_winner_trace=True)
            prediction = tokenizer.decode(ids, skip_special_tokens=True)
            paper.atomic_json(output_path, {"dataset": dataset, "qid": qid,
                "prediction": prediction, "answers": q["answers"],
                "subspan_em": paper.subspan_em(prediction, q["answers"]),
                "generated_token_ids": ids.tolist(), "expert_winners": winners,
                "top90_pids": scores["pids"], "seconds": time.monotonic() - started})
            del cache, tokenized_prefix, tokenized_question, prior, ids
            progress.update()
    progress.complete(f"PCED predictions=500/500; {a.output_root}")


def run_no_rag(a, datasets, model, tokenizer, progress):
    import torch
    pending = []
    for dataset in paper.DATASETS:
        for q in datasets[dataset][0]:
            if not paper.result_file(a.output_root / dataset / "predictions", str(q["qid"])).is_file():
                pending.append((dataset, q, no_rag_ids(tokenizer, q["query_text"])))
    progress.stage(2, 500, initial=500 - len(pending), unit="q")
    for start in range(0, len(pending), a.batch_size):
        batch = pending[start:start + a.batch_size]
        input_ids, attention_mask = make_batch(batch, tokenizer.pad_token_id, model.device)
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for index, (dataset, q, tokens) in enumerate(batch):
            generated = output[index, input_ids.shape[1]:].tolist()
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            paper.atomic_json(paper.result_file(a.output_root / dataset / "predictions", str(q["qid"])), {
                "dataset": dataset, "qid": str(q["qid"]), "prediction": prediction,
                "answers": q["answers"], "subspan_em": paper.subspan_em(prediction, q["answers"]),
                "generated_token_ids": generated, "input_tokens": len(tokens)})
            progress.update()
        del input_ids, attention_mask, output
    progress.complete(f"No-RAG predictions=500/500; {a.output_root}")


def summarize(a, datasets, manifest):
    scores = {}
    for dataset in paper.DATASETS:
        records = [json.loads(paper.result_file(a.output_root / dataset / "predictions",
                                                str(q["qid"])).read_text())
                   for q in datasets[dataset][0]]
        correct = sum(record["subspan_em"] for record in records)
        scores[dataset] = {"n": 100, "correct": correct, "subspan_em": correct / 100,
                           "missing": 0}
    correct = sum(value["correct"] for value in scores.values())
    summary = {"arm": a.arm, "n": 500, "correct": correct,
               "micro_subspan_em": correct / 500,
               "macro5_subspan_em": sum(value["subspan_em"] for value in scores.values()) / 5,
               "datasets": scores, "manifest": str(manifest)}
    paper.atomic_json(a.output_root / "summary.json", summary)
    return summary


def main(argv=None):
    a = arguments(argv)
    progress = Progress(["reference contract", "load frozen Llama", f"{a.arm} 500q", "Subspan EM summary"])
    progress.weights[:] = [1., 3., None, 1.]
    try:
        progress.stage(0, 1)
        datasets, model_path, expected, manifest = preflight(a)
        cached = sum(paper.result_file(a.output_root / dataset / "predictions", str(q["qid"])).is_file()
                     for dataset, (queries, _) in datasets.items() for q in queries)
        progress.complete(f"same 5x100 LOFT test questions; cached={cached}/500")
        progress.log(f"[plan] GPU1 only; arm={a.arm}; fixed Top-90 caches for PCED; corrected amateur, unchanged expert prompts; greedy 32 tokens; cached={cached}, remaining={500-cached}; no gold in generation")
        if a.dry_run:
            progress.log(f"[dry-run OK] CPU-only; no model loaded or output written; root={a.output_root}")
            return
        if not manifest.exists():
            paper.atomic_json(manifest, expected)
        progress.stage(1, 1)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        sample = datasets[paper.DATASETS[0]][0][0]["query_text"]
        original_prefixes, original_question = paper.prompt_inputs(tokenizer, sample, ["probe"])
        revised_prefixes, revised_questions = prompt_inputs(tokenizer, sample, ["probe"])
        if (revised_prefixes[1] != original_prefixes[1] or
                revised_questions[1] != original_question or
                len(tokenizer(revised_questions[0], add_special_tokens=False)["input_ids"]) >
                len(tokenizer(revised_questions[1], add_special_tokens=False)["input_ids"])):
            raise RuntimeError("Expert prompt changed or amateur question pads expert stream")
        if cached < 500:
            import torch
            paper.gpu_gate(40)
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(42)
            torch.manual_seed(42)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        else:
            model = None
        progress.complete(f"frozen Llama; cached={cached}/500; backend=sdpa")
        if a.arm == "pced":
            run_pced(a, datasets, model, tokenizer, progress)
        else:
            run_no_rag(a, datasets, model, tokenizer, progress)
        progress.stage(3, 1)
        summary = summarize(a, datasets, manifest)
        progress.complete(f"{a.output_root/'summary.json'}")
        progress.log(f"[result] {a.arm} {summary['correct']}/500; root={a.output_root}")
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/4; {exc}; durable={a.output_root}; identical command resumes compatible predictions")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

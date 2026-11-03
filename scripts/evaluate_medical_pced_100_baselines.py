#!/usr/bin/env python3
"""Matched No-RAG, CE-top1 Single and Top-90 All for medical PCED pilots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medrag.training.graph_input_progress import Progress
from scripts import run_loft_pced_paper as paper
from scripts import run_medical_pced_100 as medical
from scripts.evaluate_loft_corpus_in_context import vanilla_prompt
from scripts.evaluate_loft_pced_corrected_amateur import AMATEUR_SYSTEM, AMATEUR_QUESTION


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=medical.DATASETS, required=True)
    p.add_argument("--arm", choices=("no-rag", "single", "all", "all-ranked"), required=True)
    p.add_argument("--reference-root", type=Path,
                   default=ROOT / "results/medical_pced_seeded100_v1")
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "results/medical_pced_seeded100_v1_baselines")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.reference_root, a.output_root = a.reference_root.resolve(), a.output_root.resolve()
    a.batch_size = a.batch_size or (1 if a.arm in ("all", "all-ranked") else 16)
    if a.batch_size < 1 or a.reference_root == a.output_root:
        p.error("Invalid batch size or output root")
    return a


def cohort(a):
    reference = a.reference_root / a.dataset
    manifest_path = reference / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare PCED cohort first: {manifest_path}")
    ref = json.loads(manifest_path.read_text())
    if ref.get("version") != medical.VERSION or ref.get("questions") != 100 or ref.get("top_k") != 90:
        raise RuntimeError("Incompatible medical PCED reference manifest")
    selection = argparse.Namespace(dataset=a.dataset, fetch_k=ref["fetch_k"],
                                   sample_seed=ref["sample_seed"])
    _, _, corpus, questions = medical.selected_inputs(selection)
    if [str(q["qid"]) for q in questions] != ref["sample_qids"]:
        raise RuntimeError("Medical sample changed since PCED preparation")
    by_pid = {str(row["pid"]): row for row in corpus}
    rankings = {}
    if a.arm != "no-rag":
        for q in questions:
            qid = str(q["qid"])
            rank_path = paper.result_file(reference / "rankings", qid)
            if not rank_path.is_file():
                raise FileNotFoundError(f"Missing fixed PCED ranking: {rank_path}")
            rank = json.loads(rank_path.read_text())
            selected = rank["top90"]
            if rank["qid"] != qid or len(selected) != 90 or any(
                    x["pid"] not in by_pid for x in selected):
                raise RuntimeError(f"Invalid Top-90 for {qid}")
            rankings[qid] = rank
    contract = {"version": "medical_pced_baselines_seeded100_v1", "dataset": a.dataset,
                "arm": a.arm, "question_qids": ref["sample_qids"],
                "reference_manifest_sha256": paper.sha(manifest_path),
                "ranking_sha256": {qid: paper.sha(paper.result_file(reference / "rankings", qid))
                                   for qid in rankings},
                "llama": ref["llama"], "top_k": 90,
                "single_selection": "CE first among the fixed Top-90" if a.arm == "single" else None,
                "all_order": ("numpy default_rng(125)" if a.arm == "all" else
                              "CE Top-90 order" if a.arm == "all-ranked" else None),
                "prompt": "own-knowledge No-RAG" if a.arm == "no-rag" else "official vanilla document prompt",
                "max_new_tokens": 32, "do_sample": False, "batch_size": a.batch_size,
                "metric": "paper Subspan EM", "script_sha256": paper.sha(Path(__file__))}
    folder = a.output_root / a.dataset / a.arm
    old = folder / "manifest.json"
    if old.exists():
        previous = json.loads(old.read_text())
        # Adding the ranked-order arm changes this source file without
        # changing the completed no-rag/single/shuffled-all computations.
        comparable = {k: v for k, v in contract.items() if k != "script_sha256"}
        prior_comparable = {k: v for k, v in previous.items() if k != "script_sha256"}
        if previous != contract and not (a.arm in ("no-rag", "single", "all")
                                         and prior_comparable == comparable):
            raise RuntimeError(f"Incompatible baseline resume contract: {old}")
    return ref, questions, by_pid, rankings, folder, contract


def make_prompt(tokenizer, a, q, rank, by_pid, rng):
    if a.arm == "no-rag":
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": AMATEUR_SYSTEM},
             {"role": "user", "content": AMATEUR_QUESTION.format(question=q["query_text"])}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        ) + paper.ANSWER_PREFIX, []
    selected = rank["top90"]
    pids = ([selected[0]["pid"]] if a.arm == "single" else
            [selected[int(i)]["pid"] for i in rng.permutation(90)] if a.arm == "all" else
            [item["pid"] for item in selected])
    return vanilla_prompt(tokenizer, q["query_text"],
                          [paper.format_doc(by_pid[pid]) for pid in pids]), pids


def main():
    import numpy as np
    a = arguments()
    progress = Progress(["CPU contract/token budget", "load frozen Llama",
                         f"{a.dataset} {a.arm} generation", "Subspan EM summary"])
    progress.weights[:] = [1., 3., None, 1.]
    try:
        progress.stage(0, 1)
        ref, questions, by_pid, rankings, folder, contract = cohort(a)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ref["llama"], local_files_only=True,
                                                  padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        config = json.loads((Path(ref["llama"]) / "config.json").read_text())
        rng = np.random.default_rng(125)
        prompts = []
        for q in questions:
            qid = str(q["qid"])
            prompt, pids = make_prompt(tokenizer, a, q, rankings.get(qid), by_pid, rng)
            count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if count + 32 > int(config["max_position_embeddings"]):
                raise RuntimeError(f"Prompt exceeds Llama context: {qid}, tokens={count}")
            prompts.append((q, prompt, pids, count))
        cached = sum(paper.result_file(folder / "predictions", str(q["qid"])).is_file()
                     for q in questions)
        progress.complete(f"sample=100, cached={cached}, max_prompt_tokens={max(x[3] for x in prompts)}")
        if a.dry_run:
            progress.log("[dry-run OK] No output written or GPU allocated")
            return
        if not (folder / "manifest.json").exists():
            paper.atomic_json(folder / "manifest.json", contract)
        pending = [x for x in prompts if not paper.result_file(
            folder / "predictions", str(x[0]["qid"])).exists()]
        progress.stage(1, 1)
        if pending:
            import torch
            from transformers import AutoModelForCausalLM
            paper.gpu_gate(40)
            torch.backends.cuda.enable_cudnn_sdp(False)
            random.seed(42)
            torch.manual_seed(42)
            model = AutoModelForCausalLM.from_pretrained(
                ref["llama"], local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda:0").eval()
            model.requires_grad_(False)
        progress.complete(f"frozen Llama; pending={len(pending)}")
        progress.stage(2, 100, initial=cached, unit="q")
        offset, batch_size = 0, a.batch_size
        while offset < len(pending):
            block = pending[offset:offset+batch_size]
            try:
                inputs = tokenizer([x[1] for x in block], add_special_tokens=False,
                                   return_tensors="pt", padding=True).to(model.device)
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=32,
                                            pad_token_id=tokenizer.pad_token_id, use_cache=True)
            except torch.cuda.OutOfMemoryError:
                if "inputs" in locals():
                    del inputs
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                progress.log(f"[memory] retry with batch_size={batch_size}")
                continue
            for j, (q, _, pids, count) in enumerate(block):
                ids = output[j, inputs["input_ids"].shape[1]:].tolist()
                pred = tokenizer.decode(ids, skip_special_tokens=True)
                paper.atomic_json(paper.result_file(folder / "predictions", str(q["qid"])), {
                    "qid": str(q["qid"]), "dataset": a.dataset, "arm": a.arm,
                    "prediction": pred, "answers": q["answers"],
                    "subspan_em": paper.subspan_em(pred, q["answers"]),
                    "generated_token_ids": ids, "document_pids": pids, "prompt_tokens": count})
                progress.update()
            offset += len(block)
            del inputs, output
        progress.complete(f"predictions=100; {folder/'predictions'}")
        progress.stage(3, 1)
        rows = [json.loads(paper.result_file(folder / "predictions", str(q["qid"])).read_text())
                for q in questions]
        correct = sum(row["subspan_em"] for row in rows)
        paper.atomic_json(folder / "summary.json", {"dataset": a.dataset, "arm": a.arm,
                         "n": 100, "correct": correct, "subspan_em": correct / 100,
                         "missing": 0, "manifest": str(folder / "manifest.json")})
        progress.complete(f"{correct}/100; {folder/'summary.json'}")
        progress.log(f"[result] {a.dataset} {a.arm} Subspan EM={correct}/100")
    except BaseException as exc:
        progress.log(f"[FAILED] stage={progress.index+1}/4, item={progress.done}/{progress.total}; "
                     f"{exc}; durable={a.output_root}; rerun same command to resume")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()

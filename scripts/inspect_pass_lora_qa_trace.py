#!/usr/bin/env python3
"""Inspect saved expert decisions, optionally replay one full-vocabulary logit matrix on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--arm", default="pass_pced")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--qid", help="Question ID; default first saved record in the requested dataset")
    p.add_argument("--step", type=int, default=0, help="Zero-based generated token decision")
    p.add_argument("--full-vocab-out", type=Path, help="Optional CPU-only reconstructed logits NPZ for this single step")
    a = p.parse_args()
    import hashlib
    import numpy as np
    from transformers import AutoTokenizer
    from medrag.generation.pced_trained_pass import MODES
    contract = json.loads((a.run_root / "manifest.json").read_text())["contract"]
    folder = a.run_root / "reasoning" / a.arm / a.dataset
    candidates = sorted(folder.glob("*.json")) if a.qid is None else [folder / (hashlib.sha256(a.qid.encode()).hexdigest()+".json")]
    if not candidates or not candidates[0].is_file():
        raise FileNotFoundError(f"No matching record in {folder}")
    record = json.loads(candidates[0].read_text())
    inputs = json.loads((a.run_root / "inputs" / a.dataset / candidates[0].name).read_text())
    tok = AutoTokenizer.from_pretrained(contract["model"], local_files_only=True)
    with np.load(a.run_root / record["trace"], allow_pickle=False) as data:
        t = a.step
        if not 0 <= t < len(data["token_ids"]):
            raise ValueError(f"step must be 0..{len(data['token_ids'])-1}")
        begin, end = data["candidate_offsets"][t:t+2]
        ids, logits = data["candidate_ids"][begin:end], data["candidate_logits"][begin:end].T
        selected = int(data["token_ids"][t])
        selected_col = int(np.flatnonzero(ids == selected)[0])
        beta, log_prior = float(data["beta"]), data["log_prior"]
        report = {"dataset": a.dataset, "qid": record["qid"], "question": inputs["question"],
            "prefix": tok.decode(record["reasoning_token_ids"][:t]), "step": t,
            "selected_id": selected, "selected_text": tok.decode([selected]), "winner": int(data["winner"][t]),
            "mode": MODES[int(data["mode"][t])], "beta": beta, "experts": []}
        if "counter_token" in data:
            for name in ("counter", "no_prior", "no_contrast"):
                report[name] = {"token": tok.decode([int(data[name+"_token"][t])]),
                                "winner": int(data[name+"_winner"][t])}
        for k in range(logits.shape[0]):
            top_ids = data["raw_top_ids"][t, k]
            row = {"stream": k, "document_pid": inputs["pids"][k-1] if k else None,
                "raw_top": [{"id": int(v), "text": tok.decode([int(v)]), "logit": float(z)}
                            for v, z in zip(top_ids, data["raw_top_logits"][t, k])],
                "selected_token_logit": float(logits[k, selected_col]),
                "selected_token_rank": int(data["selected_token_rank"][t, k])}
            if k and "pass_log_probability" in data:
                raw = float(logits[k, selected_col])
                contrast = beta * (raw - float(logits[0, selected_col]))
                prior = contract["decoder"]["gamma"] * float(log_prior[k-1])
                penalty = contract["decoder"]["pass_weight"] * float(data["log_not_pass"][t, k-1])
                row.update(p_pass=float(np.exp(data["pass_log_probability"][t, k-1])),
                    active=bool(data["active"][t, k-1]), selected_token_components={"raw": raw,
                        "contrast": contrast, "retrieval_prior": prior, "pass_penalty": penalty,
                        "ungated_pced": raw+contrast+prior, "gated_pced_if_active": raw+contrast+prior+penalty})
            report["experts"].append(row)
        if a.full_vocab_out:
            if a.full_vocab_out.exists():
                raise FileExistsError(a.full_vocab_out)
            import torch
            from safetensors import safe_open
            torch.set_num_threads(4)
            model = Path(contract["model"])
            index = json.loads((model / "model.safetensors.index.json").read_text())
            shard = model / index["weight_map"]["lm_head.weight"]
            with safe_open(shard, framework="pt", device="cpu") as sf:
                weight = sf.get_tensor("lm_head.weight")
            hidden = torch.from_numpy(data["hidden_bf16"][t].copy()).view(torch.bfloat16)
            # CPU FP32 accumulation; exact saved candidate logits remain the source of truth.
            replay = torch.nn.functional.linear(hidden.float(), weight.float()).detach().numpy()
            error = float(np.abs(replay[:, ids] - logits).max())
            a.full_vocab_out.parent.mkdir(parents=True, exist_ok=True)
            with a.full_vocab_out.open("xb") as fh:
                np.savez_compressed(fh, logits=replay, qid=record["qid"], step=t)
            report["full_vocab_replay"] = {"path": str(a.full_vocab_out.resolve()),
                "max_absolute_error_vs_saved_candidates": error,
                "note": "CPU accumulation can differ from actual BF16 GPU LM-head output; exact candidate logits above are authoritative."}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

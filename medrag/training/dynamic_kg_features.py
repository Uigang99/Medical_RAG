"""Frozen MedCPT windows for a dynamic controller; no answer/option features."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from medrag.training.graph_input_data import atomic_json, digest, sha_file


def article_texts(spec):
    return spec["node_texts"] + spec["proposition_texts"] + [e[3] for e in spec["edges"]]


def token_windows(tokenizer, text, max_length=512):
    """Non-overlapping content windows, each with its own special tokens.

Every original content token is retained. Encoder windowing does not preserve
cross-window contextualization; it is not lossless semantic encoding.
"""
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if special != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Expected the MedCPT single-sequence BERT [CLS] ... [SEP] template")
    capacity = max_length - special
    if capacity < 1: raise ValueError("No space for encoder content")
    # Transformers 5 tokenizers no longer expose build_inputs_with_special_tokens.
    # MedCPT is explicitly BERT, so preserve its validated single-text template.
    return [[tokenizer.cls_token_id] + ids[i:i+capacity] + [tokenizer.sep_token_id]
            for i in range(0, max(1, len(ids)), capacity)]


def feature_plan(layouts):
    return {"query": sorted({x["question_text"] for x in layouts}),
            "article": sorted({s for x in layouts for s in article_texts(x)})}


def encode_bank(root, plans, models, contract_hash, progress, indices, batch_size, torch_save, load_torch):
    """256-text durable shards. Encoders unload before Llama is loaded."""
    from transformers import AutoModel, AutoTokenizer
    root = Path(root)
    for kind in ("query", "article"):
        texts = plans[kind]
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        cached = 0
        for start in range(0, len(texts), 256):
            p = directory / f"{start:07d}.pt"
            m = directory / f"{start:07d}.json"
            if m.exists():
                marker = json.loads(m.read_text())
                if marker["contract_sha256"] != contract_hash or marker["sha256"] != sha_file(p):
                    raise RuntimeError(f"Feature shard changed: {p}")
                if marker["texts_sha256"] != digest(texts[start:start+256]):
                    raise RuntimeError("Feature shard input order changed")
                cached += len(texts[start:start+256])
        progress.stage(indices[kind], len(texts), initial=cached, unit="text", kind="encoding")
        if cached == len(texts):
            progress.complete(f"{kind} features reused; {directory}")
            continue
        progress.log(f"[load] frozen MedCPT {kind}; batch={batch_size}; every content token windowed")
        tokenizer = AutoTokenizer.from_pretrained(models[kind], local_files_only=True)
        model = AutoModel.from_pretrained(models[kind], local_files_only=True, dtype=torch.float32).to("cuda:0").eval()
        model.requires_grad_(False)
        for start in range(0, len(texts), 256):
            path = directory / f"{start:07d}.pt"
            marker_path = path.with_suffix(".json")
            if marker_path.exists(): continue
            part = texts[start:start+256]
            windows, offsets = [], [0]
            for text in part:
                windows.extend(token_windows(tokenizer, text)); offsets.append(len(windows))
            vectors, cursor = [], 0
            while cursor < len(windows):
                size = min(batch_size, len(windows)-cursor)
                while True:
                    try:
                        inputs = tokenizer.pad({"input_ids": windows[cursor:cursor+size]}, padding=True,
                                               return_tensors="pt").to("cuda:0")
                        with torch.no_grad():
                            v = model(**inputs).last_hidden_state[:, 0].float()
                        if not torch.isfinite(v).all(): raise FloatingPointError("Nonfinite MedCPT features")
                        vectors.append(v.cpu().to(torch.float16)); del inputs, v
                        break
                    except torch.cuda.OutOfMemoryError:
                        if "inputs" in locals(): del inputs
                        if "v" in locals(): del v
                        gc.collect(); torch.cuda.empty_cache()
                        if size == 1: raise
                        size = max(1, size//2)
                        progress.log(f"[OOM] MedCPT retry batch={size}; next batch restores {batch_size}")
                cursor += size
            torch_save(path, {"texts": part, "offsets": offsets, "vectors": torch.cat(vectors),
                              "contract_sha256": contract_hash})
            atomic_json(marker_path, {"contract_sha256": contract_hash, "sha256": sha_file(path),
                                     "texts_sha256": digest(part), "windows": len(windows)})
            progress.update(len(part))
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        progress.complete(f"frozen {kind} features complete; {directory}")
    atomic_json(root / "COMPLETE.json", {"contract_sha256": contract_hash,
                "texts": {k: len(v) for k, v in plans.items()}, "answer_or_option_features": False})
    return FeatureBank.load(root, plans, contract_hash, load_torch)


class FeatureBank:
    def __init__(self, vectors, lookup):
        self.vectors, self.lookup = vectors, lookup

    @classmethod
    def load(cls, root, plans, contract_hash, load_torch):
        vectors, lookup = {}, {}
        for kind in ("query", "article"):
            blocks, index, cursor = [], {}, 0
            for start in range(0, len(plans[kind]), 256):
                path = Path(root) / kind / f"{start:07d}.pt"
                marker = json.loads(path.with_suffix(".json").read_text())
                if marker["contract_sha256"] != contract_hash or marker["sha256"] != sha_file(path):
                    raise RuntimeError(f"Invalid feature shard: {path}")
                item = load_torch(path)
                if (item["contract_sha256"] != contract_hash or item["texts"] != plans[kind][start:start+256]
                        or item["vectors"].shape[1] != 768 or not torch.isfinite(item["vectors"]).all()):
                    raise RuntimeError("Feature tensor/contract mismatch")
                for text, a, b in zip(item["texts"], item["offsets"][:-1], item["offsets"][1:], strict=True):
                    if b <= a: raise RuntimeError("Missing encoder window")
                    index[text] = (cursor+a, cursor+b)
                blocks.append(item["vectors"]); cursor += len(item["vectors"])
            vectors[kind], lookup[kind] = torch.cat(blocks), index
        return cls(vectors, lookup)

    def for_layout(self, spec, device):
        a, b = self.lookup["query"][spec["question_text"]]
        q = self.vectors["query"][a:b].to(device=device, dtype=torch.float32)
        pieces, groups = [], []
        for i, text in enumerate(article_texts(spec)):
            a, b = self.lookup["article"][text]
            pieces.append(self.vectors["article"][a:b]); groups.extend([i]*(b-a))
        return q, torch.cat(pieces).to(device=device, dtype=torch.float32), torch.tensor(groups, device=device)

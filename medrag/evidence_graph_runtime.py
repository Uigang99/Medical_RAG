"""Local, resumable KEV classification; no generation and no API charges."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

from medrag.training.graph_input_data import atomic_json, digest

KEV_MAX_STATE = 7552
KEV_MAX_ROW = 8192
# cuDNN SDPA cannot build a plan for some padded Qwen3.5 decision batches
# on the current H200 stack. Keep fused alternatives and a math fallback;
# do not change the model's mask, precision, or decision inputs.
KEV_SDPA_BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH")


def localize_checkpoint(checkpoint, base):
    """Validate the released family, then use its local tokenizer/weight paths."""
    if checkpoint.full:
        if checkpoint.meta.base != "Qwen/Qwen3.8-27B":
            raise RuntimeError(f"Unvalidated full-weight KEV base: {checkpoint.meta.base}")
        # Full weights are loaded from checkpoint.path, never from the 9B base.
        # Checkpoint.load resolves the tokenizer through meta.base as well.
        checkpoint.meta.base = str(Path(checkpoint.path).resolve())
    else:
        if checkpoint.meta.base != "Qwen/Qwen3.5-9B-Base":
            raise RuntimeError(f"Unvalidated KEV base: {checkpoint.meta.base}")
        checkpoint.meta.base = str(Path(base).resolve())
    checkpoint.meta.base_revision = None


class DecisionCache:
    def __init__(self, root, contract_hash):
        self.root, self.contract_hash = Path(root), contract_hash

    def path(self, job):
        return self.root / job["id"][:2] / (job["id"] + ".json")

    def get(self, job):
        path = self.path(job)
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        if (result.get("contract_hash") != self.contract_hash
                or result.get("request_hash") != digest(job["body"])):
            raise RuntimeError(f"Incompatible decision cache: {path}")
        if set(result["answers"]) != set(job["body"]["questions"]):
            raise RuntimeError(f"Malformed cached decision: {path}")
        for key, value in result["answers"].items():
            scores = value["probabilities"]
            if (set(scores) != set(job["body"]["questions"][key]["criteria"])
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())
                    or abs(sum(scores.values()) - 1) > 1e-4):
                raise RuntimeError(f"Invalid cached probabilities: {path}")
        return result

    def put(self, job, answers, **extra):
        payload = {"contract_hash": self.contract_hash, "request_hash": digest(job["body"]),
                   "request": job, "answers": answers, **extra}
        atomic_json(self.path(job), payload)
        return payload


class LocalDecisions:
    """One KEV checkpoint; batch limit counts decision rows, not requests.

The full-row forward path intentionally avoids the hybrid prefix-cache path
that failed in the earlier route experiment. Each request may have several
independent fixed-choice questions.
"""
    def __init__(self, source, adapter, base, cache, batch_rows=32):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this process; run the wrapper on H200 GPU1")
        sys.path.insert(0, str(source))
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        checkpoint = Checkpoint(adapter)
        localize_checkpoint(checkpoint, base)
        self.tokenizer, self.model = checkpoint.load(
            "cuda:0", LoadOptions(dtype=torch.bfloat16, merge=True, fused=False, cuda_graphs=False))
        self.model.eval()
        self.request_type, self.to_record = SystemOneRequest, to_record
        self.cache, self.batch_rows = cache, batch_rows
        self.parity = None
        self.batch_disabled = False
        self.oom_splits = 0

    def encode(self, job):
        record, meta = self.to_record(self.request_type.model_validate(job["body"]))
        encoded = self.model.encode(self.tokenizer, record, max_state=KEV_MAX_STATE,
                                    max_branch=KEV_MAX_ROW, strict=True)
        if encoded["state_truncated"]:
            raise RuntimeError(f"KEV context truncation forbidden: {job['id']}")
        return encoded, meta

    def _forward(self, batch):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if self.batch_disabled and len(batch) > 1:
            return [output for item in batch for output in self._forward([item])]
        try:
            # Scope this to KEV, including single-request parity forwards.
            # Restore the caller's backend settings even if a forward fails.
            with torch.inference_mode(), sdpa_kernel(
                    [getattr(SDPBackend, name) for name in KEV_SDPA_BACKENDS]):
                output = self.model.forward_batch([x[1] for x in batch], shared_prefix=False)
                if self.parity is None and len(batch) > 1:
                    single = self.model.forward(batch[0][1])
                    deltas = [float((torch.softmax(a, -1) - torch.softmax(b, -1)).abs().max())
                              for a, b in zip(single, output[0], strict=True)]
                    passed = all(torch.allclose(torch.softmax(a, -1), torch.softmax(b, -1), atol=.02, rtol=.02)
                                 and int(a.argmax()) == int(b.argmax())
                                 for a, b in zip(single, output[0], strict=True))
                    self.parity = {"max_probability_difference": max(deltas), "passed": passed,
                                   "policy": "use single-request forwards if batch parity fails"}
                    if not passed:
                        self.batch_disabled = True
                        del output
                        return [x for item in batch for x in self._forward([item])]
                return [[torch.softmax(z, -1).float().cpu().tolist() for z in record] for record in output]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.oom_splits += 1
            if len(batch) <= 1:
                raise RuntimeError("KEV single request does not fit; no decision was committed for it") from None
            middle = len(batch) // 2
            return self._forward(batch[:middle]) + self._forward(batch[middle:])

    def score(self, jobs, stage, emit):
        unique = {j["id"]: j for j in jobs}
        results = {key: self.cache.get(j) for key, j in unique.items()}
        pending = [j for key, j in unique.items() if results[key] is None]
        total = sum(len(j["body"]["questions"]) for j in unique.values())
        cached = total - sum(len(j["body"]["questions"]) for j in pending)
        emit("stage", index=stage, total=max(1, total), initial=cached if total else 1, unit="decision", kind="kev")
        encoded = [(j, *self.encode(j)) for j in pending]
        encoded.sort(key=lambda x: (len(x[1]["ids"]), x[0]["id"]))
        batches, batch, count = [], [], 0
        for item in encoded:
            rows = len(item[2])
            if batch and count + rows > self.batch_rows:
                batches.append(batch)
                batch, count = [], 0
            batch.append(item)
            count += rows
        if batch:
            batches.append(batch)
        for batch in batches:
            start = time.monotonic()
            outputs = self._forward(batch)
            for (job, enc, meta), scores in zip(batch, outputs, strict=True):
                answers = {}
                for m, values in zip(meta, scores, strict=True):
                    probabilities = dict(zip(m["keys"], values, strict=True))
                    if (not all(math.isfinite(v) for v in values) or abs(sum(values) - 1) > 1e-4):
                        raise RuntimeError("KEV emitted nonfinite/non-normalized scores")
                    winner = max(probabilities, key=probabilities.get)
                    answers[m["id"]] = {"choice": winner, "confidence": probabilities[winner],
                                        "probabilities": probabilities}
                results[job["id"]] = self.cache.put(job, answers, state_tokens=enc["state_tokens"],
                                                    batch_size=len(batch), batch_seconds=time.monotonic()-start)
                emit("update", amount=len(meta))
        emit("complete", message=f"{len(unique)} requests; {cached}/{total} cached decisions; durable={self.cache.root}")
        return results

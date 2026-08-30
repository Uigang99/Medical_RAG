"""Frozen semantic-filter features for learned document attention.

The binary semantic checkpoint is used exactly as it was trained: one official
RAG2 evidence/question prompt and one decoder decision step.  In addition to
the two label logits, this module returns an evidence-token masked mean of the
bidirectional Flan-T5 encoder state.  The feature is therefore independent for
each question-document pair and cannot leak preceding documents from a causal
multi-document prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from .rag2_official import OFFICIAL_INSTRUCTION, resolve_configured_label_token_ids


EVIDENCE_PREFIX = f"{OFFICIAL_INSTRUCTION}\n\nEvidence: "
QUESTION_MARKER = "\n\nQuestion: "
POOLING_VERSION = "official_evidence_token_masked_mean_full_input_v2"


def official_evidence_span(prompt: str) -> tuple[int, int]:
    """Return the character span occupied by Evidence in an official prompt."""

    if not prompt.startswith(EVIDENCE_PREFIX):
        raise ValueError("official_filter_input_prefix_mismatch")
    end = prompt.find(QUESTION_MARKER, len(EVIDENCE_PREFIX))
    if end <= len(EVIDENCE_PREFIX):
        raise ValueError("official_filter_input_evidence_span_invalid")
    return len(EVIDENCE_PREFIX), end


@dataclass(frozen=True)
class SemanticFeatureBatch:
    features: torch.Tensor
    logits: torch.Tensor
    margins: torch.Tensor
    probabilities: torch.Tensor
    evidence_token_counts: torch.Tensor


class FrozenSemanticEvidenceEncoder:
    """Extract frozen binary semantic logits and 1024d evidence vectors."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_input_length: int = 1280,
        batch_size: int = 64,
    ) -> None:
        if max_input_length <= 0 or batch_size <= 0:
            raise ValueError("max_input_length and batch_size must be positive")
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.max_input_length = int(max_input_length)
        self.batch_size = int(batch_size)
        self.active_batch_size = self.batch_size
        logging.info("Loading frozen semantic feature encoder: %s", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Semantic evidence pooling requires a fast tokenizer with offsets")
        model_dtype = dtype if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=model_dtype,
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        expected_config = {
            "rag2_filter_label_mode": "semantic_binary",
            "rag2_filter_input_format": "rag2_official_evidence_question_v1",
            "rag2_filter_decision_rule": "first_decoder_step_label_token_softmax",
        }
        mismatches = {
            name: {"expected": expected, "actual": getattr(self.model.config, name, None)}
            for name, expected in expected_config.items()
            if getattr(self.model.config, name, None) != expected
        }
        if mismatches:
            raise ValueError(f"Semantic checkpoint contract mismatch: {mismatches}")
        names, token_ids = resolve_configured_label_token_ids(self.tokenizer, self.model.config)
        if tuple(names) != ("helpful", "not helpful"):
            raise ValueError(f"Expected a binary semantic checkpoint, found labels={names}")
        self.label_token_ids = token_ids
        self.decoder_start_token_id = int(self.model.config.decoder_start_token_id)
        self.hidden_size = int(self.model.config.d_model)
        logging.info(
            "Frozen semantic feature encoder ready: device=%s hidden=%d max_length=%d",
            self.device,
            self.hidden_size,
            self.max_input_length,
        )

    def _encode_batch(self, prompts: list[str]) -> SemanticFeatureBatch:
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        if bool((lengths > self.max_input_length).any()):
            offending = [
                int(index)
                for index in torch.nonzero(lengths > self.max_input_length, as_tuple=False)
                .flatten()
                .tolist()
            ]
            raise RuntimeError(
                "Semantic prompt exceeds the full-input budget; refusing to truncate away "
                f"question/options: max={self.max_input_length} rows={offending[:8]} "
                f"lengths={[int(lengths[index]) for index in offending[:8]]}"
            )
        offsets = encoded.pop("offset_mapping")
        evidence_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        for index, prompt in enumerate(prompts):
            evidence_start, evidence_end = official_evidence_span(prompt)
            row_offsets = offsets[index]
            evidence_mask[index] = (
                (row_offsets[:, 1] > evidence_start)
                & (row_offsets[:, 0] < evidence_end)
                & (row_offsets[:, 1] > row_offsets[:, 0])
            )
        counts = evidence_mask.sum(dim=1)
        if bool((counts == 0).any()):
            raise RuntimeError("Semantic evidence pooling produced an empty token mask")

        inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        decoder_input_ids = torch.full(
            (len(prompts), 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            encoder_output = self.model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            hidden = encoder_output.last_hidden_state
            mask = evidence_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            decoded = self.model(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            label_ids = [
                self.label_token_ids["helpful"],
                self.label_token_ids["not helpful"],
            ]
            logits = decoded.logits[:, 0, label_ids].float()
            probabilities = torch.softmax(logits, dim=-1)
            margins = logits[:, 0] - logits[:, 1]
        return SemanticFeatureBatch(
            features=pooled.to(device="cpu", dtype=torch.float16),
            logits=logits.to(device="cpu", dtype=torch.float32),
            margins=margins.to(device="cpu", dtype=torch.float32),
            probabilities=probabilities.to(device="cpu", dtype=torch.float32),
            evidence_token_counts=counts.to(device="cpu", dtype=torch.int16),
        )

    def encode_prompts(
        self,
        prompts: Sequence[str],
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> SemanticFeatureBatch:
        """Encode prompts with OOM-aware batching while preserving input order."""

        output: list[SemanticFeatureBatch] = []
        start = 0
        while start < len(prompts):
            size = min(self.active_batch_size, len(prompts) - start)
            try:
                batch = self._encode_batch(list(prompts[start : start + size]))
            except torch.OutOfMemoryError:
                if self.device.type != "cuda" or size <= 1:
                    raise
                torch.cuda.empty_cache()
                self.active_batch_size = max(1, size // 2)
                logging.warning(
                    "Semantic feature OOM; retrying with batch_size=%d",
                    self.active_batch_size,
                )
                continue
            output.append(batch)
            start += size
            if progress_callback is not None:
                progress_callback(size)
        if not output:
            empty = torch.empty((0, self.hidden_size), dtype=torch.float16)
            return SemanticFeatureBatch(
                features=empty,
                logits=torch.empty((0, 2), dtype=torch.float32),
                margins=torch.empty((0,), dtype=torch.float32),
                probabilities=torch.empty((0, 2), dtype=torch.float32),
                evidence_token_counts=torch.empty((0,), dtype=torch.int16),
            )
        return SemanticFeatureBatch(
            features=torch.cat([item.features for item in output], dim=0),
            logits=torch.cat([item.logits for item in output], dim=0),
            margins=torch.cat([item.margins for item in output], dim=0),
            probabilities=torch.cat([item.probabilities for item in output], dim=0),
            evidence_token_counts=torch.cat(
                [item.evidence_token_counts for item in output], dim=0
            ),
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

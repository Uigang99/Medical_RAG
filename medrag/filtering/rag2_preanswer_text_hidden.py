from __future__ import annotations

"""Inference for the RAG2 pre-answer text+hidden document filter.

The label used to train this filter was derived with a gold-answer direction,
but inference deliberately extracts only h0 and hD-h0.  No answer key, gold
direction, projection score, or answer transition enters this module.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from ..core import BenchmarkSample, RetrievedDocument
from .rag2_filter import build_filter_input_from_evidence
from .rag2_official import LABEL_NAMES, add_label_tokens, resolve_label_token_ids


PREANSWER_PROMPT_VERSION = "rag2_fixed_direct_choice_context_v1"
FILTER_ARCHITECTURE_VERSION = "rag2_hidden_feature_filter_ablation_v1"
FINAL_ANSWER_PREFILL = "Final answer:"
CHOICES = ("A", "B", "C", "D")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _evidence(document: RetrievedDocument, max_doc_chars: int) -> str:
    value = _clean(document.text or document.title)
    if max_doc_chars > 0 and len(value) > max_doc_chars:
        value = value[: max_doc_chars - 3].rstrip() + "..."
    if not value:
        raise ValueError(f"Empty evidence for document {document.stable_id}")
    return value


def build_preanswer_user_prompt(
    sample: BenchmarkSample,
    document_text: str | None,
) -> str:
    if not isinstance(sample.options, dict) or any(choice not in sample.options for choice in CHOICES):
        raise ValueError(f"Pre-answer hidden extraction requires A/B/C/D options: {sample.id}")
    options_text = "\n".join(f"{choice}. {sample.options[choice]}" for choice in CHOICES)
    context = document_text.strip() if document_text and document_text.strip() else "None"
    return (
        "Select the single best answer to the following medical multiple-choice question.\n"
        "Output exactly one uppercase option letter from the given options.\n"
        "Do not provide an explanation or any additional text.\n\n"
        f"Question:\n{sample.question.strip()}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Context:\n{context}"
    )


def _render_input_ids(tokenizer: Any, user_prompt: str, marker_ids: Sequence[int]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False)) + list(marker_ids)


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(value) for value in sequences)
    input_ids = torch.full(
        (len(sequences), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, -length:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, -length:] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


class HiddenPrefixSeq2Seq(nn.Module):
    """Exact text+hidden architecture used by the training script."""

    main_input_name = "input_ids"

    def __init__(
        self,
        backbone: Any,
        hidden_size: int,
        prefix_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.generation_config = getattr(backbone, "generation_config", None)
        self.input_mode = "text_hidden"
        d_model = int(backbone.config.d_model)
        self.h0_norm = nn.LayerNorm(hidden_size)
        self.h0_projection = nn.Linear(hidden_size, d_model)
        self.delta_projection = nn.Linear(hidden_size, d_model, bias=False)
        self.delta_magnitude = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        self.prefix_type_embedding = nn.Parameter(torch.empty(2, d_model))
        nn.init.normal_(self.prefix_type_embedding, mean=0.0, std=0.02)
        self.prefix_dropout = nn.Dropout(float(prefix_dropout))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h0: torch.Tensor,
        delta_h: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> Any:
        h0_float = h0.float()
        delta_float = delta_h.float()
        magnitude = torch.linalg.vector_norm(delta_float, dim=-1, keepdim=True).clamp_min(1e-6)
        delta_unit = delta_float / magnitude
        h0_token = self.h0_projection(self.h0_norm(h0_float))
        delta_token = self.delta_projection(delta_unit) + self.delta_magnitude(torch.log1p(magnitude))
        prefix = torch.stack((h0_token, delta_token), dim=1)
        prefix = self.prefix_dropout(prefix + self.prefix_type_embedding.unsqueeze(0))
        text_embeddings = self.backbone.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat((prefix.to(text_embeddings.dtype), text_embeddings), dim=1)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        complete_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        return self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=complete_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )


class PreAnswerLayerExtractor:
    """Extract h0 and hD without constructing the gold-derived direction c."""

    def __init__(
        self,
        model_path: Path,
        *,
        layer: int,
        batch_size: int,
        max_input_tokens: int,
        device: str,
        dtype: str,
        attn_implementation: str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for hidden extraction but is unavailable")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map[dtype]
        self.batch_size = max(1, int(batch_size))
        self.max_input_tokens = int(max_input_tokens)
        self.layer = int(layer)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.marker_ids = self.tokenizer.encode(FINAL_ANSWER_PREFILL, add_special_tokens=False)
        logging.info("Loading pre-answer state model: %s", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=attn_implementation,
        )
        blocks = int(self.model.config.num_hidden_layers)
        if not 1 <= self.layer < blocks:
            raise ValueError(f"Hidden layer must be in [1,{blocks - 1}], got {self.layer}")
        self.hidden_size = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval().to(self.device)
        logging.info(
            "Pre-answer state model ready: device=%s layer=%s hidden=%s prompt=%s",
            self.device,
            self.layer,
            self.hidden_size,
            PREANSWER_PROMPT_VERSION,
        )

    def _encode(self, samples: Sequence[BenchmarkSample], contexts: Sequence[str | None]) -> list[list[int]]:
        sequences: list[list[int]] = []
        for sample, context in zip(samples, contexts):
            value = _render_input_ids(
                self.tokenizer,
                build_preanswer_user_prompt(sample, context),
                self.marker_ids,
            )
            if len(value) > self.max_input_tokens:
                raise ValueError(
                    f"Pre-answer prompt exceeds {self.max_input_tokens} tokens for {sample.id}: {len(value)}"
                )
            if value[-len(self.marker_ids) :] != self.marker_ids:
                raise RuntimeError("Final-answer marker is not the prompt suffix")
            sequences.append(value)
        return sequences

    def _forward(self, sequences: Sequence[Sequence[int]]) -> torch.Tensor:
        input_ids, attention_mask, position_ids = _pad_sequences(
            sequences, self.tokenizer.pad_token_id, self.device
        )
        with torch.inference_mode():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[self.layer][:, -1, :].float().cpu()
        return hidden

    def states(self, samples: Sequence[BenchmarkSample], contexts: Sequence[str | None]) -> torch.Tensor:
        if len(samples) != len(contexts):
            raise ValueError("Sample/context length mismatch")
        values: list[torch.Tensor] = []
        active = self.batch_size
        start = 0
        while start < len(samples):
            batch_samples = samples[start : start + active]
            batch_contexts = contexts[start : start + active]
            try:
                values.append(self._forward(self._encode(batch_samples, batch_contexts)))
            except torch.OutOfMemoryError:
                if active <= 1:
                    raise
                torch.cuda.empty_cache()
                active = max(1, active // 2)
                logging.warning("Hidden extraction OOM; retrying with batch_size=%s", active)
                continue
            start += len(batch_samples)
        return torch.cat(values, dim=0)

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TextHiddenRag2Filter:
    """Score question-document pairs with text plus target-LLM state change."""

    def __init__(
        self,
        checkpoint_path: Path,
        backbone_path: Path,
        state_model_path: Path,
        *,
        layer: int = 28,
        hidden_batch_size: int = 64,
        filter_batch_size: int = 128,
        max_hidden_input_tokens: int = 2048,
        max_filter_input_length: int = 768,
        max_doc_chars: int = 0,
        helpful_threshold: float = 0.5,
        device: str = "cuda:0",
        bf16: bool = True,
        hidden_dtype: str = "bfloat16",
        hidden_attn_implementation: str = "eager",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.filter_batch_size = max(1, int(filter_batch_size))
        self.max_filter_input_length = int(max_filter_input_length)
        self.max_doc_chars = int(max_doc_chars)
        self.helpful_threshold = float(helpful_threshold)
        self.use_bf16 = bool(bf16 and self.device.type == "cuda")
        if not 0.0 <= self.helpful_threshold <= 1.0:
            raise ValueError("Helpful threshold must be in [0,1]")
        architecture_path = checkpoint_path / "rag2_hidden_filter_architecture.json"
        if architecture_path.exists():
            import json

            architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
            if architecture.get("input_mode") != "text_hidden":
                raise RuntimeError(f"Expected text_hidden checkpoint: {checkpoint_path}")
        weight_path = checkpoint_path / "pytorch_model.bin"
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)

        self.state_extractor = PreAnswerLayerExtractor(
            state_model_path,
            layer=layer,
            batch_size=hidden_batch_size,
            max_input_tokens=max_hidden_input_tokens,
            device=device,
            dtype=hidden_dtype,
            attn_implementation=hidden_attn_implementation,
        )
        logging.info("Loading text+hidden filter: %s", checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path), use_fast=True, local_files_only=True
        )
        backbone = AutoModelForSeq2SeqLM.from_pretrained(str(backbone_path), local_files_only=True)
        add_label_tokens(self.tokenizer, backbone)
        self.model = HiddenPrefixSeq2Seq(backbone, hidden_size=self.state_extractor.hidden_size)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Text+hidden checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        del state
        gc.collect()
        # Training kept master weights in float32 and used bf16 autocast.
        # Mirror that contract instead of permanently casting LayerNorm and
        # the newly learned projection layers to bf16.
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.label_token_ids = resolve_label_token_ids(self.tokenizer)
        logging.info("Text+hidden filter ready: device=%s threshold=%.4f", self.device, self.helpful_threshold)

    def _score_batch(
        self,
        samples: Sequence[BenchmarkSample],
        evidences: Sequence[str],
        h0: torch.Tensor,
        hD: torch.Tensor,
    ) -> list[dict[str, float | str]]:
        prompts = [
            build_filter_input_from_evidence(sample, evidence, 0, input_format="official")
            for sample, evidence in zip(samples, evidences)
        ]
        encoded = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_filter_input_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        decoder_start = int(self.model.config.decoder_start_token_id)
        decoder_input_ids = torch.full(
            (len(samples), 1), decoder_start, dtype=torch.long, device=self.device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                h0=h0.to(self.device),
                delta_h=(hD - h0).to(self.device),
                decoder_input_ids=decoder_input_ids,
            )
            token_ids = [self.label_token_ids["helpful"], self.label_token_ids["not helpful"]]
            logits = outputs.logits[:, 0, token_ids].float()
            probabilities = torch.softmax(logits, dim=-1)
        rows: list[dict[str, float | str]] = []
        for index in range(len(samples)):
            probability = float(probabilities[index, 0].cpu())
            margin = float((logits[index, 0] - logits[index, 1]).cpu())
            prediction = "helpful" if probability >= self.helpful_threshold else "not helpful"
            rows.append(
                {
                    "prediction": prediction,
                    "score_helpful": float(logits[index, 0].cpu()),
                    "score_not_helpful": float(logits[index, 1].cpu()),
                    "margin": margin,
                    "prob_helpful": probability,
                }
            )
        return rows

    def score_documents(
        self,
        samples: list[BenchmarkSample],
        candidate_lists: list[list[RetrievedDocument]],
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[list[RetrievedDocument]]:
        if len(samples) != len(candidate_lists):
            raise ValueError("Sample/candidate-list length mismatch")
        # h0 is shared by every document for one question and is computed once.
        h0_all = self.state_extractor.states(samples, [None] * len(samples))
        flat_samples: list[BenchmarkSample] = []
        flat_documents: list[RetrievedDocument] = []
        flat_question_rows: list[int] = []
        flat_evidences: list[str] = []
        for question_row, (sample, documents) in enumerate(zip(samples, candidate_lists)):
            for document in documents:
                flat_samples.append(sample)
                flat_documents.append(document)
                flat_question_rows.append(question_row)
                flat_evidences.append(_evidence(document, self.max_doc_chars))

        active = self.filter_batch_size
        start = 0
        while start < len(flat_documents):
            batch_samples = flat_samples[start : start + active]
            batch_evidences = flat_evidences[start : start + active]
            question_rows = flat_question_rows[start : start + active]
            try:
                hD = self.state_extractor.states(batch_samples, batch_evidences)
                h0 = h0_all[question_rows]
                scores = self._score_batch(batch_samples, batch_evidences, h0, hD)
            except torch.OutOfMemoryError:
                if active <= 1:
                    raise
                torch.cuda.empty_cache()
                active = max(1, active // 2)
                logging.warning("Text+hidden filter OOM; retrying with batch_size=%s", active)
                continue
            for document, score in zip(flat_documents[start : start + active], scores):
                document.filter_prediction = str(score["prediction"])
                document.filter_score = float(score["margin"])
                document.filter_prob_helpful = float(score["prob_helpful"])
                document.metadata["preanswer_text_hidden_filter"] = {
                    "prompt_version": PREANSWER_PROMPT_VERSION,
                    "hidden_layer": self.state_extractor.layer,
                    "hidden_inputs": ["h0", "delta_h=hD-h0"],
                    "forbidden_inputs": ["gold_answer", "c", "projection_score", "answer_transition"],
                    "helpful_threshold": self.helpful_threshold,
                }
            if progress_callback is not None:
                progress_callback(len(batch_samples))
            start += len(batch_samples)
        return candidate_lists

    def close(self) -> None:
        self.state_extractor.close()
        self.model.to("cpu")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

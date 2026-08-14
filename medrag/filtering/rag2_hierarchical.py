from __future__ import annotations

"""Frozen Direct Sentence/Window Filter + learned document-sequence Transformer."""

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ..core import BenchmarkSample, RetrievedDocument
from .rag2_filter import build_filter_input_from_evidence
from .rag2_official import OFFICIAL_INSTRUCTION, resolve_label_token_ids
from .rag2_windowing import sentence_context_windows, windowing_contract


SCHEMA_VERSION = "rag2_document_transformer_v1"
EVIDENCE_PREFIX = f"{OFFICIAL_INSTRUCTION}\n\nEvidence: "
QUESTION_MARKER = "\n\nQuestion: "


class DocumentTransformer(nn.Module):
    """Architecture shared by training and MCQ inference checkpoints."""

    def __init__(
        self,
        embedding_dim: int,
        auxiliary_mean: torch.Tensor,
        auxiliary_std: torch.Tensor,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.register_buffer("auxiliary_mean", auxiliary_mean.reshape(1, 1, 3))
        self.register_buffer("auxiliary_std", auxiliary_std.reshape(1, 1, 3))
        self.input_projection = nn.Linear(embedding_dim + 3, d_model)
        self.document_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.document_token, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

    def forward(
        self,
        embeddings: torch.Tensor,
        auxiliary: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_embedding = self.embedding_norm(embeddings.float())
        normalized_auxiliary = (auxiliary.float() - self.auxiliary_mean) / self.auxiliary_std
        windows = self.input_projection(torch.cat([normalized_embedding, normalized_auxiliary], dim=-1))
        document_token = self.document_token.expand(windows.shape[0], -1, -1)
        sequence = torch.cat([document_token, windows], dim=1)
        document_mask = torch.zeros(padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device)
        complete_mask = torch.cat([document_mask, padding_mask], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=complete_mask)
        return self.classifier(self.output_norm(encoded[:, 0]))


def _prompt_evidence_span(prompt: str) -> tuple[int, int]:
    if not prompt.startswith(EVIDENCE_PREFIX):
        raise ValueError("official_filter_input_prefix_mismatch")
    end = prompt.find(QUESTION_MARKER, len(EVIDENCE_PREFIX))
    if end <= len(EVIDENCE_PREFIX):
        raise ValueError("official_filter_input_evidence_span_invalid")
    return len(EVIDENCE_PREFIX), end


def _position(window: dict[str, Any], document_length: int) -> float:
    midpoint = (float(window["char_start"]) + float(window["char_end"])) / 2.0
    return min(1.0, max(0.0, midpoint / max(1.0, float(document_length))))


class HierarchicalRag2DocumentFilter:
    """Infer original-document utility from all Direct Sentence/Window features."""

    def __init__(
        self,
        *,
        window_model_path: Path,
        document_checkpoint_path: Path,
        max_input_length: int = 768,
        window_batch_size: int = 128,
        document_batch_size: int = 512,
        context_sentences: int = 1,
        document_threshold: float = 0.5,
        device: str = "auto",
        bf16: bool = True,
    ) -> None:
        self.window_model_path = window_model_path.resolve()
        self.document_checkpoint_path = document_checkpoint_path.resolve()
        self.max_input_length = int(max_input_length)
        self.active_window_batch_size = int(window_batch_size)
        self.document_batch_size = int(document_batch_size)
        self.context_sentences = int(context_sentences)
        self.evidence_unit = "single_sentence" if self.context_sentences == 0 else "sentence_context_window"
        self.document_threshold = float(document_threshold)
        self.device = self._resolve_device(device)
        self.bf16 = bool(bf16 and self.device.type == "cuda")
        if not 0.0 <= self.document_threshold <= 1.0:
            raise ValueError("document_threshold must be in [0, 1]")
        if min(self.max_input_length, self.active_window_batch_size, self.document_batch_size) <= 0:
            raise ValueError("Length and batch sizes must be positive")

        self.tokenizer = AutoTokenizer.from_pretrained(self.window_model_path, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("Evidence-token pooling requires a fast tokenizer")
        dtype = torch.bfloat16 if self.bf16 else None
        self.window_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.window_model_path,
            local_files_only=True,
            dtype=dtype,
        )
        self.window_model.to(self.device)
        self.window_model.eval()
        self.label_token_ids = resolve_label_token_ids(self.tokenizer)
        self.decoder_start_token_id = self.window_model.config.decoder_start_token_id
        if self.decoder_start_token_id is None:
            raise RuntimeError("Window filter lacks decoder_start_token_id")

        checkpoint = torch.load(self.document_checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported Document Transformer checkpoint: {self.document_checkpoint_path}")
        config = checkpoint.get("model_config") or {}
        required_auxiliary = ["prob_helpful", "margin_helpful_minus_not_helpful", "relative_position"]
        if config.get("auxiliary_features") != required_auxiliary:
            raise RuntimeError(f"Document Transformer auxiliary contract mismatch: {config.get('auxiliary_features')}")
        if int(config.get("embedding_dim", -1)) != int(self.window_model.config.d_model):
            raise RuntimeError("Window encoder and Document Transformer embedding dimensions differ")
        if int(config.get("max_windows_per_document", 0)) != 0:
            raise RuntimeError("MCQ inference requires a checkpoint trained with all document windows")
        self.document_model = DocumentTransformer(
            embedding_dim=int(config["embedding_dim"]),
            auxiliary_mean=torch.tensor(config["auxiliary_mean"], dtype=torch.float32),
            auxiliary_std=torch.tensor(config["auxiliary_std"], dtype=torch.float32),
            d_model=int(config["d_model"]),
            num_heads=int(config["num_heads"]),
            num_layers=int(config["num_layers"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["dropout"]),
        )
        self.document_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.document_model.to(self.device)
        self.document_model.eval()
        if self.device.type == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        logging.info(
            "Hierarchical RAG2 filter ready: evidence_unit=%s filter_model=%s "
            "document_checkpoint=%s best_epoch=%s device=%s threshold=%.6f",
            self.evidence_unit,
            self.window_model_path,
            self.document_checkpoint_path,
            checkpoint.get("epoch"),
            self.device,
            self.document_threshold,
        )

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value != "auto":
            return torch.device(value)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def close(self) -> None:
        self.window_model.to("cpu")
        self.document_model.to("cpu")
        del self.window_model, self.document_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_window_features(
        self,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        utilities: list[torch.Tensor] = []
        evidence_counts: list[torch.Tensor] = []
        start = 0
        while start < len(prompts):
            size = min(self.active_window_batch_size, len(prompts) - start)
            try:
                embedding, utility, counts = self._extract_window_batch(list(prompts[start : start + size]))
            except torch.OutOfMemoryError:
                if self.device.type != "cuda" or size <= 1:
                    raise
                torch.cuda.empty_cache()
                self.active_window_batch_size = max(1, size // 2)
                logging.warning(
                    "Hierarchical window feature OOM; retrying with filter_batch_size=%s",
                    self.active_window_batch_size,
                )
                continue
            embeddings.append(embedding)
            utilities.append(utility)
            evidence_counts.append(counts)
            start += size
        return torch.cat(embeddings), torch.cat(utilities), torch.cat(evidence_counts)

    def _extract_window_batch(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        lengths = encoded["attention_mask"].sum(dim=1)
        if int(lengths.max()) > self.max_input_length:
            raise RuntimeError("Internal error: audited overlength window reached the encoder")
        evidence_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        for index, prompt in enumerate(prompts):
            evidence_start, evidence_end = _prompt_evidence_span(prompt)
            row_offsets = offsets[index]
            evidence_mask[index] = (
                (row_offsets[:, 1] > evidence_start)
                & (row_offsets[:, 0] < evidence_end)
                & (row_offsets[:, 1] > row_offsets[:, 0])
            )
        counts = evidence_mask.sum(dim=1)
        if bool((counts == 0).any()):
            raise RuntimeError("Evidence-token pooling produced an empty mask")
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        decoder_input_ids = torch.full(
            (len(prompts), 1),
            int(self.decoder_start_token_id),
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            encoder_output = self.window_model.get_encoder()(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                return_dict=True,
            )
            hidden = encoder_output.last_hidden_state
            mask = evidence_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            decoded = self.window_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                attention_mask=model_inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            label_ids = [self.label_token_ids["helpful"], self.label_token_ids["not helpful"]]
            logits = decoded.logits[:, 0, label_ids].float()
            probabilities = torch.softmax(logits, dim=-1)
            margin = logits[:, 0] - logits[:, 1]
            utility = torch.stack([logits[:, 0], logits[:, 1], probabilities[:, 0], margin], dim=-1)
        return (
            pooled.to(device="cpu", dtype=torch.float16),
            utility.to(device="cpu", dtype=torch.float32),
            counts.to(device="cpu", dtype=torch.int16),
        )

    @torch.inference_mode()
    def _score_document_sequences(
        self,
        embeddings: list[torch.Tensor],
        auxiliaries: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_out: list[torch.Tensor] = []
        probabilities_out: list[torch.Tensor] = []
        for start in range(0, len(embeddings), self.document_batch_size):
            batch_embeddings = embeddings[start : start + self.document_batch_size]
            batch_auxiliary = auxiliaries[start : start + self.document_batch_size]
            max_windows = max(int(value.shape[0]) for value in batch_embeddings)
            hidden_size = int(batch_embeddings[0].shape[1])
            padded_embeddings = torch.zeros(len(batch_embeddings), max_windows, hidden_size, dtype=torch.float16)
            padded_auxiliary = torch.zeros(len(batch_embeddings), max_windows, 3, dtype=torch.float32)
            padding_mask = torch.ones(len(batch_embeddings), max_windows, dtype=torch.bool)
            for index, (embedding, auxiliary) in enumerate(zip(batch_embeddings, batch_auxiliary, strict=True)):
                length = int(embedding.shape[0])
                padded_embeddings[index, :length] = embedding
                padded_auxiliary[index, :length] = auxiliary
                padding_mask[index, :length] = False
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.bf16,
            ):
                logits = self.document_model(
                    padded_embeddings.to(self.device),
                    padded_auxiliary.to(self.device),
                    padding_mask.to(self.device),
                )
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
            logits_out.append(logits.float().cpu())
            probabilities_out.append(probabilities.cpu())
        return torch.cat(logits_out), torch.cat(probabilities_out)

    def score_question_batch(
        self,
        samples: Sequence[BenchmarkSample],
        candidate_lists: Sequence[list[RetrievedDocument]],
        *,
        max_doc_chars: int = 0,
    ) -> int:
        """Score all documents and mutate their standard filter fields."""

        if len(samples) != len(candidate_lists):
            raise ValueError("samples/candidate_lists length mismatch")
        records: list[dict[str, Any]] = []
        flat_prompts: list[str] = []
        for sample, documents in zip(samples, candidate_lists, strict=True):
            for document in documents:
                evidence = " ".join(str(document.text or document.title or "").split())
                if max_doc_chars > 0 and len(evidence) > max_doc_chars:
                    evidence = evidence[: max_doc_chars - 3].rstrip() + "..."
                windows = sentence_context_windows(evidence, context_sentences=self.context_sentences)
                if self.context_sentences == 0 and any(
                    int(window.get("sentence_count", 0)) != 1 for window in windows
                ):
                    raise RuntimeError(
                        f"Single-sentence inference produced a non-singleton unit: {document.stable_id}"
                    )
                prompts = [
                    build_filter_input_from_evidence(sample, window["text"], 0, input_format="official")
                    for window in windows
                ]
                lengths = (
                    [
                        len(value)
                        for value in self.tokenizer(
                            prompts,
                            add_special_tokens=True,
                            truncation=False,
                            padding=False,
                        )["input_ids"]
                    ]
                    if prompts
                    else []
                )
                invalid_reason = None
                if not windows:
                    invalid_reason = "no_windows"
                elif any(length > self.max_input_length for length in lengths):
                    invalid_reason = "overlength_document"
                record = {
                    "sample": sample,
                    "document": document,
                    "evidence": evidence,
                    "windows": windows,
                    "prompts": prompts,
                    "input_lengths": lengths,
                    "invalid_reason": invalid_reason,
                    "flat_start": len(flat_prompts),
                }
                if invalid_reason is None:
                    flat_prompts.extend(prompts)
                record["flat_end"] = len(flat_prompts)
                records.append(record)

        if flat_prompts:
            flat_embeddings, flat_utility, flat_evidence_counts = self._extract_window_features(flat_prompts)
        else:
            hidden = int(self.window_model.config.d_model)
            flat_embeddings = torch.empty(0, hidden, dtype=torch.float16)
            flat_utility = torch.empty(0, 4, dtype=torch.float32)
            flat_evidence_counts = torch.empty(0, dtype=torch.int16)

        valid_records: list[dict[str, Any]] = []
        sequence_embeddings: list[torch.Tensor] = []
        sequence_auxiliary: list[torch.Tensor] = []
        for record in records:
            document = record["document"]
            if record["invalid_reason"] is not None:
                document.filter_prediction = "not helpful"
                document.filter_prob_helpful = 0.0
                document.filter_score = -1.0e9
                document.metadata["window_filter"] = {
                    "aggregation": "document_transformer",
                    "evidence_unit": self.evidence_unit,
                    "threshold": self.document_threshold,
                    "failure": record["invalid_reason"],
                    "scored_window_count": 0,
                    "raw_helpful_window_count": 0,
                    "window_decisions": [],
                }
                continue
            start, end = int(record["flat_start"]), int(record["flat_end"])
            embeddings = flat_embeddings[start:end]
            utility = flat_utility[start:end]
            positions = torch.tensor(
                [_position(window, len(record["evidence"])) for window in record["windows"]],
                dtype=torch.float32,
            )
            auxiliary = torch.stack([utility[:, 2], utility[:, 3], positions], dim=-1)
            record["utility"] = utility
            record["positions"] = positions
            record["evidence_counts"] = flat_evidence_counts[start:end]
            valid_records.append(record)
            sequence_embeddings.append(embeddings)
            sequence_auxiliary.append(auxiliary)

        if valid_records:
            document_logits, document_probabilities = self._score_document_sequences(
                sequence_embeddings, sequence_auxiliary
            )
            for record, logits, probability in zip(
                valid_records, document_logits, document_probabilities, strict=True
            ):
                document = record["document"]
                utility = record["utility"]
                decisions = []
                for index, window in enumerate(record["windows"]):
                    decisions.append(
                        {
                            "window_id": window["window_id"],
                            "centre_sentence_id": window["centre_sentence_id"],
                            "sentence_ids": window["sentence_ids"],
                            "sentence_count": int(window["sentence_count"]),
                            "char_start": int(window["char_start"]),
                            "char_end": int(window["char_end"]),
                            "window_sha256": window["sha256"],
                            "input_length": int(record["input_lengths"][index]),
                            "evidence_token_count": int(record["evidence_counts"][index]),
                            "normalized_position": float(record["positions"][index]),
                            "prediction": "helpful" if float(utility[index, 2]) >= 0.5 else "not helpful",
                            "score_helpful": float(utility[index, 0]),
                            "score_not_helpful": float(utility[index, 1]),
                            "margin_helpful_minus_not_helpful": float(utility[index, 3]),
                            "prob_helpful_over_candidates": float(utility[index, 2]),
                        }
                    )
                probability_value = float(probability)
                margin_value = float(logits[1] - logits[0])
                document.filter_prob_helpful = probability_value
                document.filter_score = margin_value
                document.filter_prediction = (
                    "helpful" if probability_value >= self.document_threshold else "not helpful"
                )
                document.metadata["window_filter"] = {
                    "aggregation": "document_transformer",
                    "evidence_unit": self.evidence_unit,
                    "threshold": self.document_threshold,
                    "windowing": windowing_contract(self.context_sentences),
                    "scored_window_count": len(decisions),
                    "raw_helpful_window_count": sum(item["prediction"] == "helpful" for item in decisions),
                    "document_logit_not_helpful": float(logits[0]),
                    "document_logit_helpful": float(logits[1]),
                    "document_logit_margin": margin_value,
                    "document_prob_helpful": probability_value,
                    "window_decisions": decisions,
                }
        return len(records)

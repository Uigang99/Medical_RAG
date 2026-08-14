from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core import GenerationOutput, PromptRequest
from .base import TextGenerator


def _strip_chat_template_thinking_stub(rendered: str) -> str:
    return re.sub(
        r"(?is)(<\|im_start\|>assistant\s*)<think>\s*$",
        r"\1",
        rendered,
    )


def _render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return _strip_chat_template_thinking_stub(str(rendered))


def _silence_noisy_loggers() -> None:
    for name in (
        "vllm",
        "vllm.engine",
        "vllm.executor",
        "vllm.worker",
        "vllm.config",
        "vllm.entrypoints",
        "transformers",
        "huggingface_hub",
        "sentence_transformers",
        "tokenizers",
        "accelerate",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


class TransformersChatGenerator(TextGenerator):
    def __init__(
        self,
        model_path: Path,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        logging.info("Loading LLM with transformers: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        logging.info("Transformers LLM ready.")

    def _render_prompt(self, request: PromptRequest) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return _render_chat_prompt(self.tokenizer, request.messages)
        return request.rendered

    def generate_batch(self, requests: list[PromptRequest]) -> list[GenerationOutput]:
        outputs: list[GenerationOutput] = []
        do_sample = self.temperature > 0
        for request in requests:
            prompt = self._render_prompt(request)
            encoded = self.tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                generated = self.model.generate(**encoded, **generate_kwargs)
            new_tokens = generated[0, encoded["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(GenerationOutput(text=text, prompt=prompt, raw_text=text))
        return outputs


class VLLMChatGenerator(TextGenerator):
    def __init__(
        self,
        model_path: Path,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        bad_words: list[str] | None = None,
        use_chat_template: bool = True,
        use_tqdm: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int | None = 8192,
        gdn_prefill_backend: str | None = None,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        performance_mode: str = "throughput",
        max_num_seqs: int | None = 128,
        max_num_batched_tokens: int | None = 65536,
        enable_prefix_caching: bool | None = True,
        use_flashinfer_sampler: bool = False,
        assistant_prefill: str | None = None,
        allowed_token_ids: list[int] | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = max_model_len
        self.use_chat_template = bool(use_chat_template)
        self.use_tqdm = bool(use_tqdm)
        self.assistant_prefill = str(assistant_prefill or "")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1" if use_flashinfer_sampler else "0")
        _silence_noisy_loggers()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        self._llm_cls = LLM
        self._llm_kwargs: dict[str, Any] = {
            "model": str(model_path),
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": "bfloat16",
            "disable_log_stats": True,
            "runner": "generate",
            "enforce_eager": enforce_eager,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "performance_mode": performance_mode,
        }
        if max_num_seqs is not None:
            self._llm_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            self._llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if enable_prefix_caching is not None:
            self._llm_kwargs["enable_prefix_caching"] = enable_prefix_caching
        if max_model_len is not None:
            self._llm_kwargs["max_model_len"] = max_model_len
        if gdn_prefill_backend:
            self._llm_kwargs["additional_config"] = {"gdn_prefill_backend": gdn_prefill_backend}
        logging.info("Loading LLM with vLLM: %s", model_path)
        self.llm = self._init_llm_with_fallback()
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or ["<|im_end|>", "<|eot_id|>"],
            bad_words=bad_words or [],
            allowed_token_ids=allowed_token_ids,
        )
        logging.info("vLLM LLM ready.")

    def _try_init_llm(self, gpu_util: float):
        kwargs = dict(self._llm_kwargs)
        kwargs["gpu_memory_utilization"] = float(gpu_util)
        return self._llm_cls(**kwargs)

    @staticmethod
    def _parse_free_total(err: str) -> tuple[float | None, float | None]:
        # vLLM has used both of the following forms across releases:
        #   "Free memory on device (123.4/140.0 GiB) ..."
        #   "Free memory on device cuda:0 (123.4/140.0 GiB) ..."
        # Accept the optional device identifier so a recoverable, slightly too
        # high memory-utilization request actually reaches the retry below.
        match = re.search(
            r"Free memory on device(?:\s+cuda:\d+)?\s*\(([\d.]+)\s*/\s*([\d.]+)\s*GiB\)",
            err,
        )
        if not match:
            return None, None
        try:
            return float(match.group(1)), float(match.group(2))
        except Exception:
            return None, None

    def _init_llm_with_fallback(self):
        try:
            return self._try_init_llm(self.gpu_memory_utilization)
        except Exception as exc:
            free_gb, total_gb = self._parse_free_total(str(exc))
            if free_gb is None or total_gb is None or total_gb <= 0:
                raise
            safe_util = max(0.25, min(0.98, (free_gb / total_gb) - 0.03))
            if safe_util >= self.gpu_memory_utilization:
                raise
            logging.warning(
                "vLLM init retry with lower gpu_memory_utilization: %.3f -> %.3f "
                "(free=%.2fGiB total=%.2fGiB)",
                self.gpu_memory_utilization,
                safe_util,
                free_gb,
                total_gb,
            )
            self.gpu_memory_utilization = safe_util
            return self._try_init_llm(self.gpu_memory_utilization)

    def _render_prompt(self, request: PromptRequest) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            rendered = _render_chat_prompt(self.tokenizer, request.messages)
        else:
            rendered = request.rendered
        return rendered + self.assistant_prefill

    def generate_batch(self, requests: list[PromptRequest]) -> list[GenerationOutput]:
        prompts = [self._render_prompt(request) for request in requests]
        sampling_params: Any = self.sampling_params
        structured_regexes = [request.metadata.get("structured_regex") for request in requests]
        if any(structured_regexes):
            from vllm.sampling_params import StructuredOutputsParams

            per_request = []
            for regex in structured_regexes:
                params = self.sampling_params.clone()
                if regex:
                    params.structured_outputs = StructuredOutputsParams(regex=str(regex))
                per_request.append(params)
            sampling_params = per_request
        generations = self.llm.generate(prompts, sampling_params, use_tqdm=self.use_tqdm)
        outputs: list[GenerationOutput] = []
        for prompt, generation in zip(prompts, generations):
            choice = generation.outputs[0] if generation.outputs else None
            generated_text = choice.text.strip() if choice is not None else ""
            outputs.append(
                GenerationOutput(
                    text=generated_text,
                    prompt=prompt,
                    raw_text=generated_text,
                    finish_reason=getattr(choice, "finish_reason", None),
                    stop_reason=str(getattr(choice, "stop_reason", "") or "") or None,
                )
            )
        return outputs

    def generate_allowed_single_token_continuations(
        self,
        prefixes: list[str],
        allowed_strings: tuple[str, ...] = ("A", "B", "C", "D"),
    ) -> list[GenerationOutput]:
        """Greedily continue raw assistant prefixes by one allowed token."""
        from vllm import SamplingParams

        allowed_ids: list[int] = []
        for value in allowed_strings:
            ids = self.tokenizer.encode(value, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Allowed continuation {value!r} is not one token: {ids}")
            allowed_ids.append(int(ids[0]))
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            allowed_token_ids=allowed_ids,
        )
        generations = self.llm.generate(prefixes, params, use_tqdm=self.use_tqdm)
        outputs: list[GenerationOutput] = []
        for prompt, generation in zip(prefixes, generations):
            choice = generation.outputs[0] if generation.outputs else None
            generated_text = choice.text.strip() if choice is not None else ""
            outputs.append(
                GenerationOutput(
                    text=generated_text,
                    prompt=prompt,
                    raw_text=generated_text,
                    finish_reason=getattr(choice, "finish_reason", None),
                    stop_reason=str(getattr(choice, "stop_reason", "") or "") or None,
                )
            )
        return outputs

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        candidates = [
            (llm, "shutdown"),
            (llm, "close"),
            (getattr(llm, "llm_engine", None), "shutdown"),
            (getattr(llm, "llm_engine", None), "close"),
            (getattr(llm, "engine", None), "shutdown"),
            (getattr(llm, "engine", None), "close"),
        ]
        for obj, method in candidates:
            if obj is None:
                continue
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    continue
        self.llm = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

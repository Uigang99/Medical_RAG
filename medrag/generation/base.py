from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..core import GenerationOutput, PromptRequest


class TextGenerator(ABC):
    @abstractmethod
    def generate_batch(self, requests: list[PromptRequest]) -> list[GenerationOutput]:
        raise NotImplementedError


class DryRunGenerator(TextGenerator):
    def generate_batch(self, requests: list[PromptRequest]) -> list[GenerationOutput]:
        return [
            GenerationOutput(
                text="[GENERATION_SKIPPED: use --generator vllm or --generator transformers for LLM inference]",
                prompt=request.rendered,
                raw_text="[GENERATION_SKIPPED: use --generator vllm or --generator transformers for LLM inference]",
            )
            for request in requests
        ]


class LazyGenerator(TextGenerator):
    def __init__(self, factory: Callable[[], TextGenerator], name: str = "generator") -> None:
        self.factory = factory
        self.name = name
        self._generator: TextGenerator | None = None

    @property
    def generator(self) -> TextGenerator:
        if self._generator is None:
            self._generator = self.factory()
        return self._generator

    def generate_batch(self, requests: list[PromptRequest]) -> list[GenerationOutput]:
        return self.generator.generate_batch(requests)

    def close(self) -> None:
        if self._generator is None:
            return
        close_fn = getattr(self._generator, "close", None)
        if callable(close_fn):
            close_fn()
        self._generator = None

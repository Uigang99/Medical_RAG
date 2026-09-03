"""Document-position-restricted LoRA for frozen Llama K/V projections."""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch
from torch import nn


class DocumentPathLoRALinear(nn.Module):
    """Frozen Linear plus a low-rank update emitted only at document tokens."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self._document_mask: torch.Tensor | None = None
        self.last_non_document_delta_max = 0.0

    def set_document_mask(self, mask: torch.Tensor | None) -> None:
        self._document_mask = mask

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(hidden_states)
        mask = self._document_mask
        if mask is None or not bool(mask.any()):
            self.last_non_document_delta_max = 0.0
            return base
        if hidden_states.ndim != 3 or mask.shape != hidden_states.shape[:2]:
            raise RuntimeError(
                f"Document mask/input mismatch: mask={tuple(mask.shape)} hidden={tuple(hidden_states.shape)}"
            )
        raw_delta = self.lora_b(self.lora_a(self.dropout(hidden_states))) * self.scaling
        expanded = mask.to(device=raw_delta.device, dtype=raw_delta.dtype).unsqueeze(-1)
        masked_delta = raw_delta * expanded
        if bool((~mask).any()):
            self.last_non_document_delta_max = float(masked_delta[~mask].detach().abs().max().item())
        else:
            self.last_non_document_delta_max = 0.0
        return base + masked_delta


class DocumentPathAdapter:
    """Install and control K/V-only document-path adapters on Llama layers."""

    def __init__(self, model: nn.Module, *, rank: int, alpha: float, dropout: float) -> None:
        decoder = getattr(model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise TypeError("Expected a Llama-style model exposing model.layers")
        model.requires_grad_(False)
        modules: list[DocumentPathLoRALinear] = []
        for layer_index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise TypeError(f"Layer {layer_index} has no self_attn")
            for name in ("k_proj", "v_proj"):
                base = getattr(attention, name, None)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Layer {layer_index} {name} is not nn.Linear: {type(base)}")
                wrapped = DocumentPathLoRALinear(
                    base,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(attention, name, wrapped)
                modules.append(wrapped)
        if not modules:
            raise RuntimeError("No document-path adapter modules were installed")
        self.model = model
        self.modules = modules

    def set_document_mask(self, mask: torch.Tensor | None) -> None:
        for module in self.modules:
            module.set_document_mask(mask)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for module in self.modules:
            yield from module.lora_a.parameters()
            yield from module.lora_b.parameters()

    def named_trainable_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for index, module in enumerate(self.modules):
            for name, parameter in module.lora_a.named_parameters():
                yield f"adapter.{index}.lora_a.{name}", parameter
            for name, parameter in module.lora_b.named_parameters():
                yield f"adapter.{index}.lora_b.{name}", parameter

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for index, module in enumerate(self.modules):
            state[f"{index}.lora_a.weight"] = module.lora_a.weight.detach().cpu()
            state[f"{index}.lora_b.weight"] = module.lora_b.weight.detach().cpu()
        return state

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {
            key
            for index in range(len(self.modules))
            for key in (f"{index}.lora_a.weight", f"{index}.lora_b.weight")
        }
        if set(state) != expected:
            raise RuntimeError(
                f"Adapter state keys mismatch: missing={sorted(expected-set(state))[:3]} "
                f"extra={sorted(set(state)-expected)[:3]}"
            )
        for index, module in enumerate(self.modules):
            module.lora_a.weight.data.copy_(
                state[f"{index}.lora_a.weight"].to(module.lora_a.weight)
            )
            module.lora_b.weight.data.copy_(
                state[f"{index}.lora_b.weight"].to(module.lora_b.weight)
            )

    def audit(self) -> dict[str, Any]:
        trainable = list(self.named_trainable_parameters())
        unexpected = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_a" not in name and "lora_b" not in name
        ]
        return {
            "projection_modules": len(self.modules),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "unexpected_trainable_parameters": unexpected,
            "max_non_document_delta": max(
                module.last_non_document_delta_max for module in self.modules
            ),
        }

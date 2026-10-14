"""One medical-QA adapted Llama, reused for NONE/TEXT/GRAPH inputs.

No input soft prompt is trained. PEFT LoRA modifies attention/MLP projections
at every input position; it is not a document-only adapter or graph adapter.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from medrag.training.graph_input_modules import answer_losses

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def adapt_llama(base, rank=8, alpha=16):
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=alpha, lora_dropout=0., bias="none", target_modules=list(TARGET_MODULES)))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if bad or not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError(f"Unexpected trainable parameters: {bad}")
    return model


def qa_losses(model, rows, pad_id, *, graph_prompts=None):
    """Target-only CE with ordinary causal teacher forcing, no gold in prompt.

    The underlying Llama contains the installed LoRA layers. Passing it to the
    existing loss helper avoids PEFT's model wrapper changing the output type.
    """
    base = model.get_base_model()
    emb = base.get_input_embeddings()
    if graph_prompts is None:
        prompts = [emb.weight.new_empty((0, emb.embedding_dim)) for _ in rows]
        arm = "text"
    else:
        prompts, arm = graph_prompts, "graph"
    return answer_losses(base, emb, rows, prompts, arm, pad_id)


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def restore_adapter(model, state):
    expected = get_peft_model_state_dict(model)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise RuntimeError("LoRA checkpoint keys/shapes do not match this architecture")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError("Non-finite LoRA weights")
    set_peft_model_state_dict(model, state)

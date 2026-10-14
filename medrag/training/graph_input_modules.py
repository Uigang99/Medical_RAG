"""Edge-aware graph soft prompt and frozen-Llama answer loss.

G-Retriever-style Graph Transformer/mean pool/MLP, implemented with native
PyTorch scatter operations (no compiled PyG dependencies). It is an adaptation,
not a bit-identical reproduction of the upstream PyG implementation.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def training_attention_backend():
    """Keep checkpoint recomputation on the same non-cuDNN SDPA backends.

    This context must cover backward(), not just the original forward().
    Non-reentrant checkpointing recomputes attention during backward and does
    not preserve the global SDPA selection context used by the first forward.
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        yield


class EdgeAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, heads):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must divide heads")
        self.heads, self.width = heads, hidden_dim // heads
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.skip = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        n = len(x)
        src, dst = edge_index
        q = self.query(x).reshape(n, self.heads, self.width)
        k = self.key(x).reshape(n, self.heads, self.width)
        v = self.value(x).reshape(n, self.heads, self.width)
        e = self.edge(edge_attr).reshape(-1, self.heads, self.width)
        scores = (q[dst] * (k[src] + e)).sum(-1) / math.sqrt(self.width)
        # Normalize per destination/head, never across graphs.
        index = dst[:, None].expand(-1, self.heads)
        maximum = scores.new_full((n, self.heads), -torch.inf)
        maximum.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
        exp = (scores - maximum[dst]).exp()
        denom = scores.new_zeros(n, self.heads).scatter_add_(0, index, exp)
        alpha = exp / denom[dst].clamp_min(1e-12)
        msg = (v[src] + e) * alpha.unsqueeze(-1)
        result = x.new_zeros(n, self.heads, self.width)
        result.index_add_(0, dst, msg)
        return result.flatten(1) + self.skip(x)


class GraphPrompt(nn.Module):
    def __init__(self, input_dim, llm_dim, hidden_dim=1024, layers=4, heads=4):
        super().__init__()
        self.layers = nn.ModuleList(EdgeAttention(input_dim if i == 0 else hidden_dim,
                                                  hidden_dim, input_dim, heads) for i in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(layers - 1))
        self.projector = nn.Sequential(nn.Linear(hidden_dim, 2048), nn.Sigmoid(), nn.Linear(2048, llm_dim))
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        # Empty evidence gets a zero graph representation, not an invented edge.
        if not len(x):
            pooled = self.projector[0].weight.new_zeros(1, self.hidden_dim)
        else:
            h = x.float()
            for i, layer in enumerate(self.layers):
                h = layer(h, edge_index, edge_attr.float())
                if i < len(self.norms):
                    norm = self.norms[i]
                    if len(h) == 1 and self.training:
                        h = F.batch_norm(h, norm.running_mean, norm.running_var,
                                         norm.weight, norm.bias, training=False, eps=norm.eps)
                    else:
                        h = norm(h)
                    h = F.relu(h)
            pooled = h.mean(0, keepdim=True)
        return self.projector(pooled)


class TextPrompt(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.prompt = nn.Parameter(initial.detach().float().clone())

    def forward(self):
        return self.prompt


def answer_losses(llm, token_embeddings, rows, soft_prompts, arm, pad_id):
    """Exact teacher-forced CE, computing vocabulary logits at targets only.

    No generate() and no answer selection. Gold is absent from prompt IDs; gold
    prefixes appear only via ordinary causal teacher forcing. LLM weights stay
    frozen but backpropagation through its inputs must remain enabled.
    """
    inputs, labels = [], []
    for row, soft in zip(rows, soft_prompts, strict=True):
        ids = row[arm + "_prompt_ids"]
        target = row["target_ids"]
        all_ids = torch.tensor(ids + target, device=token_embeddings.weight.device)
        emb = token_embeddings(all_ids)
        soft = soft.to(emb.dtype)
        # Keep Llama-3 BOS first. Both arms insert their soft prompt here.
        packed = torch.cat((emb[:1], soft, emb[1:]), dim=0)
        inputs.append(packed)
        lab = [-100] * (len(ids) + len(soft)) + target
        labels.append(torch.tensor(lab, device=emb.device, dtype=torch.long))
    length = max(len(x) for x in inputs)
    pad = token_embeddings(torch.tensor(pad_id, device=inputs[0].device))
    masks, padded, padded_labels = [], [], []
    for emb, lab in zip(inputs, labels, strict=True):
        extra = length - len(emb)
        padded.append(torch.cat((emb, pad.expand(extra, -1))))
        padded_labels.append(F.pad(lab, (0, extra), value=-100))
        masks.append([1] * len(emb) + [0] * extra)
    mask = torch.tensor(masks, device=inputs[0].device, dtype=torch.long)
    label = torch.stack(padded_labels)
    out = llm.model(inputs_embeds=torch.stack(padded), attention_mask=mask,
                    position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                    use_cache=False, return_dict=True)
    batch, target_pos = (label != -100).nonzero(as_tuple=True)
    if bool((target_pos == 0).any()):
        raise ValueError("Target cannot occupy causal position zero")
    selected = out.last_hidden_state[batch, target_pos - 1]
    logits = llm.lm_head(selected).float()
    token_loss = F.cross_entropy(logits, label[batch, target_pos], reduction="none")
    sums = token_loss.new_zeros(len(rows)).index_add(0, batch, token_loss)
    counts = torch.bincount(batch, minlength=len(rows)).clamp_min(1)
    return sums / counts

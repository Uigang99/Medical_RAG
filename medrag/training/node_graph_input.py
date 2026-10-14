"""Relation-aware node tokens, without graph-level pooling.

The encoder is shared by every node and graph. Node IDs are bookkeeping, never
learned ID embeddings. A reverse *message* marks the receiving endpoint as the
subject of the ORIGINAL relation; it does not assert an inverse KG fact.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from medrag.training.graph_input_modules import EdgeAttention, GraphPrompt


class RoleAwareAttention(EdgeAttention):
    def __init__(self, input_dim, hidden_dim, edge_dim, heads):
        super().__init__(input_dim, hidden_dim, edge_dim, heads)
        # 0: receiver is object, 1: receiver is subject of the original edge.
        self.receiver_role = nn.Parameter(torch.empty(2, edge_dim))
        nn.init.normal_(self.receiver_role, std=0.02)

    def forward(self, x, edge_index, edge_attr):
        messages = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        attributes = torch.cat((edge_attr + self.receiver_role[0],
                                edge_attr + self.receiver_role[1]), dim=0)
        return super().forward(x, messages, attributes)


class NodeGraphPrompt(GraphPrompt):
    """Return [number_of_nodes, llm_dim] in the input node-ID order.

    Keep the previous width/depth, normalization, residuals and projector so
    this version changes the readout and (optionally) message direction, not
    the entire model. Old graph-level checkpoints are NOT reused.
    """
    def __init__(self, input_dim, llm_dim, hidden_dim=1024, layers=4, heads=4,
                 bidirectional=True):
        super().__init__(input_dim, llm_dim, hidden_dim, layers, heads)
        if bidirectional:
            self.layers = nn.ModuleList(
                RoleAwareAttention(input_dim if i == 0 else hidden_dim,
                                   hidden_dim, input_dim, heads)
                for i in range(layers))
        self.llm_dim = llm_dim
        self.bidirectional = bidirectional

    def forward(self, x, edge_index, edge_attr):
        if not len(x):
            # No fabricated entity or learned pseudo-evidence for an empty KG.
            return self.projector[0].weight.new_empty((0, self.llm_dim))
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
        return self.projector(h)  # Deliberately no h.mean(...).


def node_prompt_batch(adapter, rows, features, lookup, device):
    """Preserve per-question graph scope; never mix neighborhoods across rows."""
    result = []
    for row in rows:
        g = row["graph"]
        ni = torch.tensor([lookup[t] for t in g["node_texts"]], dtype=torch.long)
        ei = torch.tensor([lookup[t] for t in g["edge_texts"]], dtype=torch.long)
        edges = torch.tensor(g["edge_index"], dtype=torch.long, device=device)
        z = adapter(features[ni].to(device), edges, features[ei].to(device))
        if z.shape != (len(g["node_ids"]), adapter.llm_dim):
            raise RuntimeError(f"Node/token alignment failed: {row['sample_id']}")
        result.append(z)
    return result

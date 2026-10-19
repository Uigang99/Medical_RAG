"""Gold-conditioned intervention diagnostics, NOT an inference-time controller.

Only a question-local [causal KG pair, layer group] table is optimized. Llama,
text, positions and KG stay fixed. Continuous optimization is a bounded search,
not a proof of a global optimum or of generalization.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from medrag.generation.proposition_kg_attention import NAME, proposition_attention
from medrag.training.attention_intervention_diagnostic import ACTIONS, GROUPS
from medrag.training.graph_input_modules import training_attention_backend


class OracleBiasTable(nn.Module):
    def __init__(self, dim, pairs, group_width=4, groups=3):
        super().__init__()
        self.input_dim, self.layers, self.enabled = dim, group_width * groups, True
        # Wrapper device check only; this dummy weight is never optimized.
        self.output = nn.Linear(1, 1, bias=False).requires_grad_(False)
        self.register_buffer('expected_pairs', torch.tensor(pairs, dtype=torch.long).reshape(-1, 2))
        self.values = nn.Parameter(torch.zeros(len(pairs), groups, dtype=torch.float32))
        self.group_width = group_width

    def forward(self, question, propositions, pairs, roles):
        if not torch.equal(pairs, self.expected_pairs):
            raise RuntimeError('Oracle table belongs to a different ordered pair set')
        return self.values.repeat_interleave(self.group_width, dim=1)

    @torch.no_grad()
    def assign(self, values):
        values = torch.as_tensor(values, device=self.values.device, dtype=torch.float32)
        if values.shape != self.values.shape:
            raise ValueError('Oracle table shape mismatch')
        self.values.copy_(values)


def oracle_attention(module, query, key, value, attention_mask, scaling, **kwargs):
    """Use the same MATH kernel with/without gradients on controlled layers.

    Otherwise SDPA can change the forward kernel when a mask needs gradients,
    making optimization loss and evaluation loss numerically incomparable.
    Uncontrolled layers retain the existing frozen-Llama SDPA path.
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend
    contexts = kwargs.get('zero_reference_contexts')
    context = contexts[module.layer_idx] if contexts is not None else kwargs.get('proposition_context')
    if context is not None and context.matrices is not None:
        with sdpa_kernel(SDPBackend.MATH):
            return proposition_attention(module, query, key, value, attention_mask, scaling, **kwargs)
    return proposition_attention(module, query, key, value, attention_mask, scaling, **kwargs)


def install_oracle_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(NAME, oracle_attention)


def loss_and_logits(wrapper, row, pad):
    hidden = wrapper([row], pad_id=pad).last_hidden
    ids = torch.tensor(row['choice_token_ids'], device=hidden.device)
    logits = F.linear(hidden.float(), wrapper.llm.lm_head.weight[ids].float())[0]
    loss = F.cross_entropy(logits[None], torch.tensor([row['gold_index']], device=hidden.device))
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError('Nonfinite oracle logits/loss')
    return loss, logits


def outcome(loss, logits, gold):
    logits = logits.detach().float()
    others = [i for i in range(len(logits)) if i != gold]
    return dict(ce=float(loss.detach()), logits=logits.cpu().tolist(),
                p_gold=float(logits.softmax(-1)[gold]),
                margin=float(logits[gold] - logits[others].max()),
                prediction=int(logits.argmax()), gold=int(gold),
                correct=bool(int(logits.argmax()) == gold))


@torch.no_grad()
def score(wrapper, row, pad):
    with training_attention_backend():
        loss, logits = loss_and_logits(wrapper, row, pad)
    return outcome(loss, logits, row['gold_index'])


def single_table(n, pair, action):
    values = torch.zeros(n, len(GROUPS))
    group, dose = ACTIONS[action]
    values[pair, group] = dose
    return values


def single_summary(base, records, old_pairs):
    """Include no-op; keep best-CE and existence-of-correct outcomes separate."""
    def best(pool):
        choices = [dict(pair=None, action=None, score=base)] + [
            dict(pair=r['pair_index'], action=j, score=s)
            for r in pool for j, s in enumerate(r['scores'])]
        selected = min(choices, key=lambda x: x['score']['ce'])
        return dict(**selected, available_correct=any(c['score']['correct'] for c in choices),
                    best_margin=max(c['score']['margin'] for c in choices))
    old_pairs = {tuple(p) for p in old_pairs}
    return dict(sampled=best([r for r in records if tuple(r['pair']) in old_pairs]),
                exhaustive=best(records), measured_pairs=len(records),
                sampled_pairs=sum(tuple(r['pair']) in old_pairs for r in records))


def additive_candidates(base, records, n):
    """Raw top2/4/8 combined effects, not oracle-filtered after application."""
    useful = []
    for r in records:
        j = min(range(len(ACTIONS)), key=lambda j: r['scores'][j]['ce'])
        gain = base['ce'] - r['scores'][j]['ce']
        if gain > 0:
            useful.append((gain, r['pair_index'], j))
    useful.sort(key=lambda x: (-x[0], x[1], x[2]))
    result = []
    for k in (2, 4, 8):
        chosen = useful[:k]
        values = torch.zeros(n, len(GROUPS))
        for _, i, j in chosen:
            values += single_table(n, i, j)
        result.append(dict(k=k, selected=[dict(pair=i, action=j, single_gain=g) for g, i, j in chosen],
                           additive_gain=sum(g for g, _, _ in chosen), values=values))
    return result


def optimize_step(wrapper, row, pad, optimizer, bound):
    """One full CE-gradient update, only to question-local pair/group scalars."""
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    with training_attention_backend():
        loss, logits = loss_and_logits(wrapper, row, pad)
        loss.backward()
    gradient = wrapper.controller.values.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError('Oracle table gradient is missing or nonfinite')
    norm = float(gradient.norm())
    if any(p.grad is not None for p in wrapper.llm.parameters()):
        raise RuntimeError('Frozen target unexpectedly received gradients')
    before = wrapper.controller.values.detach().cpu().clone()
    optimizer.step()
    with torch.no_grad():
        wrapper.controller.values.clamp_(-bound, bound)
    step = float((wrapper.controller.values.detach().cpu() - before).norm())
    return outcome(loss, logits, row['gold_index']), before, dict(
        gradient_norm=norm, update_norm=step,
        saturation_fraction=float((wrapper.controller.values.detach().abs() >= bound - 1e-6).float().mean()))

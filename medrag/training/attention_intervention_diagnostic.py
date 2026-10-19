"""Controlled, finite-action diagnostics for proposition attention.

The target is empirical target-Llama CE reduction, not a medical relation label.
All selected pairs share the same mask schedule; zero is an explicit action.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from medrag.generation.proposition_kg_attention import (
    NAME, PropositionAttentionModel, block_bias, proposition_attention, validate_layout,
)
from medrag.training.proposition_pair_effects import stable_key, paired_interval

GROUPS = ((12, 13, 14, 15), (18, 19, 20, 21), (24, 25, 26, 27))
DOSES = (-2., -1., -.5, .5, 1., 2.)
ACTIONS = tuple((g, dose) for g in range(len(GROUPS)) for dose in DOSES)


def cohort(rows, excluded, seed, sizes):
    """Fresh, stem-disjoint official TRAIN cohorts; original Val1000 is reserved.

    Capability/dev/heldout are allocated before the nested training pool, so
    increasing train_n does not move any other split.
    """
    result, used = [], set()
    blocked = {tuple(r['input_ids'][slice(*r['question_span'])]) for r in rows
               if r['sample_id'] in excluded}
    for split in ('capability', 'dev', 'heldout', 'train'):
        n = sizes[split]
        nm = math.floor(n * .7 + .5)
        for dataset, count in (('medmcqa', nm), ('medqa', n - nm)):
            pool = sorted((r for r in rows if r['split'] == 'train' and r['dataset'] == dataset),
                          key=lambda r: stable_key(seed, dataset, r['sample_id']))
            selected = 0
            for row in pool:
                stem = tuple(row['input_ids'][slice(*row['question_span'])])
                if row['sample_id'] in excluded or stem in used or stem in blocked:
                    continue
                if selected == count:
                    break
                used.add(stem)
                result.append(dict(row, diagnostic_split=split))
                selected += 1
            if selected != count:
                raise ValueError(f'Insufficient fresh TRAIN questions: {split}/{dataset}')
    return result


def pair_metadata(row, pair):
    r, s = pair
    spans = row['proposition_spans']
    ids = row['input_ids']
    q = set(ids[slice(*row['question_span'])])
    rp, sp = (set(ids[slice(*spans[i])]) for i in (r, s))
    docs = row.get('proposition_document_ids', [''] * len(spans))
    n = max(1, len(ids))
    return [math.log1p(spans[r][1] - spans[r][0]), math.log1p(spans[s][1] - spans[s][0]),
            (spans[r][0] - spans[s][1]) / n, r / max(1, len(spans)), s / max(1, len(spans)),
            len(q & rp) / max(1, len(q | rp)), len(q & sp) / max(1, len(q | sp)),
            float(docs[r] == docs[s])]


def candidates(row, count, seed):
    """Hash sample real edges; nearest unused nonedge by transparent covariates.

    Matching is approximate, not semantic equivalence. Distances/covariates are
    retained for audit; control pairs never become original KG edges.
    """
    graph = {tuple(p): role for p, role in zip(row['pairs'], row['pair_roles'], strict=True)}
    real = sorted(graph, key=lambda p: stable_key(seed, row['sample_id'], *p))[:count]
    n = len(row['proposition_spans'])
    pool = { (r, s): pair_metadata(row, (r, s)) for r in range(n) for s in range(r)
             if (r, s) not in graph }
    items = []
    # Keep equal candidate budgets. Unmatched real edges are not swept in A/B/C.
    for pair in real:
        if not pool:
            break
        meta = pair_metadata(row, pair)
        other = min(pool, key=lambda p: (sum((a-b)**2 for a,b in zip(meta,pool[p],strict=True)),
                                        stable_key(seed, row['sample_id'], *p)))
        distance = sum((a-b)**2 for a,b in zip(meta,pool[other],strict=True))
        items.append(dict(pair=list(pair), kind='graph', roles=graph[pair], covariates=meta,
                          match_distance=distance))
        items.append(dict(pair=list(other), kind='non_graph', roles=[0]*4, covariates=pool.pop(other),
                          match_distance=distance))
    return items


def candidate_row(row, items):
    out = dict(row, pairs=[i['pair'] for i in items], pair_roles=[i['roles'] for i in items])
    validate_layout(out)
    return out


def subset(rows, n, seed):
    """Proportional two-dataset subset, without labels or correctness sampling."""
    nm = math.floor(n*.7+.5)
    return [r for ds,k in (('medmcqa',nm),('medqa',n-nm))
            for r in sorted((r for r in rows if r['dataset']==ds),
                            key=lambda r:stable_key(seed,'order',r['sample_id']))[:k]]


def reverse_propositions(row, items):
    """Reverse body token blocks, keep prompt wrappers and options identical.

    This is a token permutation diagnostic, not re-retrieval. Each position's
    [P#] wrapper stays in place; IDs/roles follow the moved source proposition.
    """
    spans, ids = row['proposition_spans'], row['input_ids']
    order = list(reversed(range(len(spans))))
    packed, new_spans, cursor = [], [], 0
    for slot, old in enumerate(order):
        a, b = spans[slot]
        packed.extend(ids[cursor:a])
        start = len(packed)
        packed.extend(ids[slice(*spans[old])])
        new_spans.append([start, len(packed)])
        cursor = b
    packed.extend(ids[cursor:])
    result = dict(row, input_ids=packed, proposition_spans=new_spans)
    for name in ('proposition_ids', 'proposition_document_ids'):
        if name in row:
            result[name] = [row[name][i] for i in order]
    reverse = {old: new for new, old in enumerate(order)}
    result['pairs'], result['pair_roles'] = [], []
    for pair, roles in zip(row['pairs'], row['pair_roles'], strict=True):
        a, b = map(reverse.__getitem__, pair)
        result['pairs'].append([max(a,b), min(a,b)])
        result['pair_roles'].append([roles[i] for i in (0,2,1,3)] if a < b else roles)
    moved = deepcopy(items)
    for item in moved:
        a,b = map(reverse.__getitem__, item['pair'])
        item['pair'] = [max(a,b), min(a,b)]
        if a < b:
            item['roles'] = [item['roles'][i] for i in (0,2,1,3)]
        item['covariates'] = pair_metadata(result,item['pair'])
    assert Counter(packed) == Counter(ids) and len(packed) == len(ids)
    validate_layout(result)
    return result, moved


class InterventionController(nn.Module):
    def __init__(self, dim, layers=4):
        super().__init__()
        self.input_dim, self.layers, self.enabled = dim, layers, True
        self.output = nn.Linear(1, 1, bias=False)
        self.actions, self.capture, self.captured = [], False, None

    def forward(self, q, p, pairs, roles):
        if self.capture:
            self.captured = dict(q=q.detach().float().cpu(), p=p.detach().float().cpu())
        out = q.new_zeros((len(pairs), self.layers), dtype=torch.float32)
        for pair, layer, dose in self.actions:
            out[pair, layer] += dose
        return out


class AttentionProbe:
    """Read actual pre-softmax QK and bias for sampled receiver query tokens.

    Observational only: delegate the attention output unchanged. Records mean
    sender-span probability mass over all heads and up to 8 receiver tokens.
    """
    def __init__(self):
        self.request = None
        self.values = []

    def __call__(self, module, query, key, value, attention_mask, scaling, **kwargs):
        if self.request is not None and module.layer_idx == self.request['layer']:
            c = kwargs.get('proposition_context')
            if c is None or c.matrices is None:
                raise RuntimeError('Attention probe requires the enabled block-mask path')
            ra, rb = self.request['receiver_span']
            sa, sb = self.request['sender_span']
            positions = torch.linspace(ra, rb-1, min(8,rb-ra), device=query.device).round().long().unique()
            q = query[0,:,positions,:].float()
            k = key[0].float().repeat_interleave(query.shape[1]//key.shape[1], dim=0)
            logits = q @ k.transpose(-1,-2) * scaling
            bias = block_bias(c, ra, rb, torch.float32)[0,0][positions-ra]
            logits = logits + bias[None]
            allowed = torch.arange(key.shape[-2],device=query.device)[None,:] <= positions[:,None]
            logits = logits.masked_fill(~allowed[None], -torch.inf)
            mass = logits.softmax(-1)[...,sa:sb].sum(-1).mean()
            self.values.append(dict(layer=module.layer_idx+1, mass=float(mass),
                                    sampled_queries=len(positions), heads=query.shape[1]))
        return proposition_attention(module,query,key,value,attention_mask,scaling,**kwargs)


def install_probe(probe):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(NAME, probe)


def features(captured, original_row, items):
    p, q = captured['p'], captured['q']
    neighbors = torch.zeros_like(p)
    degree = torch.zeros(len(p))
    for r,s in original_row['pairs']:
        neighbors[r] += p[s]; neighbors[s] += p[r]
        degree[r] += 1; degree[s] += 1
    neighbors /= degree.clamp_min(1)[:,None]
    pairs = torch.tensor([i['pair'] for i in items],dtype=torch.long).reshape(-1,2)
    # Semantic vectors: question, receiver, sender, both KG neighborhoods,
    # and the whole supplied proposition set. No option suffix or gold input.
    n = len(items)
    pooled = p.mean(0) if len(p) else q.new_zeros(q.shape)
    x = torch.stack([q.expand(n,-1), p[pairs[:,0]], p[pairs[:,1]],
                     neighbors[pairs[:,0]], neighbors[pairs[:,1]], pooled.expand(n,-1)],1)
    aux = torch.tensor([i['covariates']+i['roles'] for i in items],dtype=torch.float32).reshape(n,12)
    return dict(x=x, aux=aux)


class EffectPredictor(nn.Module):
    """Small supervised function of frozen semantic features and local KG context."""
    def __init__(self, dim, width=32, outputs=18):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(6*width+12,64), nn.GELU(), nn.Linear(64,outputs))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, f):
        device = next(self.parameters()).device
        z = self.project(f['x'].to(device)).flatten(1)
        return self.head(torch.cat((z,f['aux'].to(device)),1))


def gains(record):
    return np.asarray([[record['baselines'][g]['ce'] - score['ce']
                        for g, group in enumerate(pair) for score in group]
                       for pair in record['scores']],dtype=float).reshape(len(record['items']),len(ACTIONS))


def select_actions(predicted, items, kind, k, floor):
    """One action per distinct pair; explicit abstention; no gold is consulted."""
    scored = []
    for i,item in enumerate(items):
        if item['kind'] != kind:
            continue
        j = int(np.argmax(predicted[i]))
        if predicted[i,j] > floor:
            scored.append((float(predicted[i,j]),i,j))
    scored.sort(key=lambda x:(-x[0],x[1],x[2]))
    return [(i,j) for _,i,j in scored[:k]]


def chosen_gain(record, predicted, kind, floor):
    selected = select_actions(predicted,record['items'],kind,1,floor)
    return float(gains(record)[selected[0]]) if selected else 0.


def compare_scores(base, new, datasets):
    n = len(base)
    transitions = Counter(('C' if b['correct'] else 'W')+'->'+('C' if x['correct'] else 'W')
                          for b,x in zip(base,new,strict=True))
    delta = [float(x['correct'])-float(b['correct']) for b,x in zip(base,new,strict=True)]
    improvement = [b['ce']-x['ce'] for b,x in zip(base,new,strict=True)]
    groups = {}
    for ds in sorted(set(datasets)):
        ix = [i for i,d in enumerate(datasets) if d == ds]
        groups[ds] = dict(n=len(ix), baseline_correct=sum(base[i]['correct'] for i in ix),
                          correct=sum(new[i]['correct'] for i in ix))
    return dict(n=n, missing=0, baseline_correct=sum(b['correct'] for b in base),
                correct=sum(x['correct'] for x in new),
                baseline_accuracy=float(np.mean([b['correct'] for b in base])),
                accuracy=float(np.mean([x['correct'] for x in new])),
                accuracy_delta_pp=100*float(np.mean(delta)),
                accuracy_delta_ci95_pp=[100*x for x in paired_interval(delta)],
                baseline_ce=float(np.mean([b['ce'] for b in base])),
                ce=float(np.mean([x['ce'] for x in new])),
                ce_reduction=float(np.mean(improvement)),ce_reduction_ci95=paired_interval(improvement),
                transitions=dict(transitions),datasets=groups)

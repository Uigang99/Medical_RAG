"""Measured local attention interventions, not semantic/medical gold labels.

The target Llama and its attention implementation remain unchanged. A small
controller learns two measured CE improvements per pair/layer (decrease,
increase); zero is an explicit abstention action. No answer enters its features.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from medrag.generation.proposition_kg_attention import PropositionPairController


def stable_key(seed, *parts):
    return hashlib.sha256((str(seed) + ":" + ":".join(map(str, parts))).encode()).hexdigest()


def select_cohort(rows, seed=314159, train_n=128, dev_n=32, heldout_n=64):
    """70:30 dataset allocation, hash sampling independent of labels/accuracy/KG."""
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source sample IDs')
    result, used = [], set()
    for part, split, n in [('train', 'train', train_n), ('dev', 'validation', dev_n),
                           ('heldout', 'validation', heldout_n)]:
        mcqa = int(math.floor(n * .7 + .5))
        for dataset, count in [('medmcqa', mcqa), ('medqa', n - mcqa)]:
            pool = sorted((r for r in rows if r['split'] == split and r['dataset'] == dataset
                           and r['sample_id'] not in used),
                          key=lambda r: stable_key(seed, split, r['sample_id']))
            if len(pool) < count:
                raise ValueError(f'Insufficient {split}/{dataset} rows')
            for r in pool[:count]:
                result.append(dict(r, pilot_split=part))
                used.add(r['sample_id'])
    stems = {}
    for r in result:
        a, b = r['question_span']
        stem = tuple(r['input_ids'][a:b])
        if stem in stems and stems[stem] != r['pilot_split']:
            raise ValueError('Identical question stems cross pilot splits; select a new cohort seed')
        stems[stem] = r['pilot_split']
    return result


def sampled_pairs(row, count, seed):
    return sorted(range(len(row['pairs'])),
                  key=lambda i: stable_key(seed, row['sample_id'], *row['pairs'][i]))[:count]


def _raw_pair_logits(net, q, p, pairs, roles):
    """Same two-round architecture as the frozen v1 controller, unbounded head.

    Keep the original file byte-identical so existing run contracts stay valid.
    Regression tests compare this trunk with v1 before its output tanh.
    """
    device = net.output.weight.device
    q = net.question_projection(q.to(device=device, dtype=torch.float32))
    p = net.proposition_projection(p.to(device=device, dtype=torch.float32))
    pairs = pairs.to(device=device, dtype=torch.long).reshape(-1, 2)
    roles = roles.to(device=device, dtype=torch.float32).reshape(-1, 4)
    if not len(pairs):
        return net.output.weight.sum() * 0 + p.new_zeros((0, net.layers))
    receiver, sender = pairs.unbind(1)
    receive, send = torch.cat((receiver, sender)), torch.cat((sender, receiver))
    er = net.role_projection(torch.cat((roles, roles[:, [0, 2, 1, 3]])))
    degree = torch.bincount(receive, minlength=len(p)).clamp_min(1)[:, None]
    h = p
    for message, update in zip(net.messages, net.updates, strict=True):
        msg = message(torch.cat((h[send], h[receive], er, q.expand(len(send), -1)), -1))
        aggregate = h.new_zeros(h.shape).index_add(0, receive, msg[:, :-1] * msg[:, -1:].sigmoid()) / degree
        h = h + update(torch.cat((h, aggregate, q.expand(len(h), -1)), -1))
    z = net.readout(torch.cat((q.expand(len(pairs), -1), p[receiver], p[sender],
                              h[receiver], h[sender], net.role_projection(roles)), -1))
    return net.output(z)


def actions(effects, floor):
    """0=decrease, 1=unchanged, 2=increase; ties/weak gains abstain."""
    utilities = torch.stack((effects[..., 0], torch.zeros_like(effects[..., 0]), effects[..., 1]), -1)
    best, which = utilities.max(-1)
    return torch.where(best > floor, which, torch.ones_like(which))


class PairEffectController(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=4, delta=.5, floor=.01):
        super().__init__()
        self.network = PropositionPairController(input_dim, hidden=hidden, layers=2*layers)
        self.input_dim, self.layers = input_dim, layers
        self.delta, self.floor = float(delta), float(floor)
        self.register_buffer('effect_scale', torch.tensor(.02))
        self.register_buffer('constant_effects', torch.zeros(layers, 2))
        self.enabled, self.mode = True, 'zero'
        self.intervention = None
        self.capture, self.captured = False, None

    @property
    def output(self):
        return self.network.output

    def predict_effects(self, q, p, pairs, roles):
        return _raw_pair_logits(self.network, q, p, pairs, roles).reshape(-1, self.layers, 2) * self.effect_scale

    def forward(self, q, p, pairs, roles):
        if self.capture:
            self.captured = dict(q=q.detach().float().cpu(), p=p.detach().float().cpu(),
                                 pairs=pairs.detach().cpu(), roles=roles.detach().cpu())
        bias = q.new_zeros((len(pairs), self.layers), dtype=torch.float32)
        if self.mode == 'intervention':
            index, layer, direction = self.intervention
            if direction not in (-1, 1):
                raise ValueError('Intervention must be exactly one signed pair/layer')
            bias[index, layer] = self.delta * direction
        elif self.mode in ('learned', 'constant'):
            effects = (self.predict_effects(q, p, pairs, roles) if self.mode == 'learned'
                       else self.constant_effects.expand(len(pairs), -1, -1))
            bias = (actions(effects, self.floor).float() - 1) * self.delta
        elif self.mode != 'zero':
            raise ValueError(self.mode)
        return bias


def targets(record, floor, device):
    selected = record['selected_pairs']
    result = torch.empty((len(selected), record['layers'], 2), device=device)
    for k, item in enumerate(record['measurements']):
        result[k] = torch.tensor(item['effects'], device=device)
    # Raw outcomes remain in the audit; do not teach effects beneath the floor.
    return result.masked_fill(result.abs() <= floor, 0)


def supervision_loss(controller, record):
    f = record['features']
    prediction = controller.predict_effects(**f)[record['selected_pairs']]
    if not prediction.numel():
        return None
    expected = targets(record, controller.floor, prediction.device)
    return F.smooth_l1_loss(prediction / controller.effect_scale,
                            expected / controller.effect_scale)


def training_constants(records, layers=4, floor=.01):
    question_means, values = [], []
    for r in records:
        if r['measurements']:
            v = np.asarray([x['effects'] for x in r['measurements']], dtype=np.float64)
            question_means.append(np.where(np.abs(v)<=floor,0,v).mean(0))
            values.extend(np.abs(v).ravel())
    if not question_means:
        return np.zeros((layers, 2)), .02
    return np.mean(question_means, axis=0), max(.02, float(np.quantile(values, .9)))


@torch.no_grad()
def local_metrics(controller, records):
    """Per-question averaging; uncertainty must resample questions, not edges."""
    rows = []
    for record in records:
        if not record['measurements']:
            rows.append(dict(sample_id=record['sample_id'], graphless=True, learned_regret=0.,
                             constant_regret=0., zero_regret=0., random_regret=0.,
                             informative=0, correct_actions=0))
            continue
        device = controller.output.weight.device
        raw = torch.tensor([m['effects'] for m in record['measurements']], device=device)
        utility = torch.stack((raw[..., 0], torch.zeros_like(raw[..., 0]), raw[..., 1]), -1)
        utility = utility.masked_fill(utility.abs() <= controller.floor, 0)
        oracle = actions(raw, controller.floor)
        oracle_gain = utility.gather(-1, oracle[..., None])[..., 0]
        prediction = controller.predict_effects(**record['features'])[record['selected_pairs']]
        selected = actions(prediction, controller.floor)
        constant = actions(controller.constant_effects.expand_as(raw), controller.floor)
        chosen = utility.gather(-1, selected[..., None])[..., 0]
        const_gain = utility.gather(-1, constant[..., None])[..., 0]
        informative = oracle != 1
        decisions=[]
        for k,m in enumerate(record['measurements']):
            decisions.append(dict(pair_index=m['pair_index'],receiver=m['receiver'],sender=m['sender'],
                layers_one_based=list(range(13,13+controller.layers)),raw_effects=m['effects'],
                predicted_effects=prediction[k].cpu().tolist(),actions=selected[k].cpu().tolist(),
                constant_actions=constant[k].cpu().tolist(),oracle_actions=oracle[k].cpu().tolist()))
        rows.append(dict(sample_id=record['sample_id'], graphless=False,decisions=decisions,
                         learned_regret=float((oracle_gain-chosen).mean()),
                         constant_regret=float((oracle_gain-const_gain).mean()),
                         zero_regret=float(oracle_gain.mean()),
                         random_regret=float((oracle_gain-utility.mean(-1)).mean()),
                         informative=int(informative.sum()),
                         correct_actions=int(((selected == oracle) & informative).sum())))
    metrics = {key: float(np.mean([r[key] for r in rows])) for key in
               ('learned_regret', 'constant_regret', 'zero_regret', 'random_regret')}
    n = sum(r['informative'] for r in rows)
    metrics.update(questions=len(rows), graphless=sum(r['graphless'] for r in rows), informative=n,
                   informative_action_accuracy=sum(r['correct_actions'] for r in rows)/max(1,n))
    return metrics, rows


def signal_report(records, floor):
    positive = negative = informative = disagreements = 0
    questions = set()
    constants, scale = training_constants(records,floor=floor)
    const_actions = actions(torch.tensor(constants), floor).numpy()
    for r in records:
        for m in r['measurements']:
            a = actions(torch.tensor(m['effects']), floor).numpy()
            positive += int((a == 2).sum()); negative += int((a == 0).sum())
            informative += int((a != 1).sum())
            disagreements += int((a != const_actions).sum())
            if (a != 1).any(): questions.add(r['sample_id'])
    return dict(questions=len(records), informative_decisions=informative,
                informative_questions=len(questions), increase=positive, decrease=negative,
                differs_from_layer_constant=disagreements, effect_scale=scale,
                constant_effects=constants.tolist())


def paired_interval(deltas, seed=1234, draws=2000):
    x = np.asarray(deltas, dtype=float)
    if not len(x): raise ValueError('Empty interval cohort')
    rng = np.random.default_rng(seed)
    values = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(1)
    return [float(v) for v in np.quantile(values, [.025, .975])]

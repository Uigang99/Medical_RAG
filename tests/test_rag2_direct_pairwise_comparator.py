from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.train_rag2_direct_pairwise_comparator import (
    DirectPairDataset,
    SymmetricPairPacker,
    choice_logits,
    question_macro_loss,
    question_macro_pair_weights,
)


class _Progress:
    def set_stage(self, *_args, **_kwargs):
        return None

    def update(self, _value=1):
        return None


class _Flat:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


class _Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(char) % 251) for char in text]

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 1

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        assert token_ids_1 is None
        return list(token_ids_0) + [self.eos_token_id]


class _SequenceClassifier(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        score = input_ids.float().sum(dim=1)
        return SimpleNamespace(logits=torch.stack((score, -score), dim=1))


def test_direct_pair_dataset_keeps_only_decisive_ordered_pairs():
    rows = [
        {
            "sample_id": "q1",
            "pair_id": f"p{index}",
            "utility_target": utility,
            "no_rag_correct_audit_only": True,
        }
        for index, utility in enumerate((0.30, 0.25, -0.20))
    ]
    dataset = DirectPairDataset(_Flat(rows), 0.1, "unit", _Progress())

    assert len(dataset) == 2
    assert dataset.question_count == 1
    assert all(pair[2] >= 0.1 for pair in dataset.pairs)
    assert all(
        rows[winner]["utility_target"] > rows[loser]["utility_target"]
        for winner, loser, *_ in dataset.pairs
    )


def test_question_macro_loss_does_not_overweight_question_with_more_pairs():
    # Three semantic pairs -> six orientations. q0 owns two pairs, q1 owns one.
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 3.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    question_index = torch.tensor([0, 0, 1])

    loss = question_macro_loss(logits, labels, question_index)
    easy = torch.nn.functional.cross_entropy(torch.tensor([[3.0, 0.0]]), torch.tensor([0]))
    hard = torch.nn.functional.cross_entropy(torch.tensor([[0.0, 3.0]]), torch.tensor([0]))
    expected = (easy + hard) / 2.0
    assert torch.allclose(loss, expected)


def test_chunk_weights_preserve_exact_question_macro_mass():
    weights = question_macro_pair_weights(torch.tensor([0, 0, 1]))
    assert torch.allclose(weights, torch.tensor([0.25, 0.25, 0.50]))
    assert torch.allclose(weights[:2].sum(), weights[2:].sum())


def test_symmetric_pair_packer_never_exceeds_budget():
    packer = SymmetricPairPacker(_Tokenizer(), max_tokens=512, minimum_document_tokens=16)
    ids = packer.pack(
        question="short question",
        options="A) one\nB) two",
        no_rag_answer="(A) one",
        document_a="a" * 1000,
        document_b="b" * 1000,
    )
    assert len(ids) == 512
    assert ids[-1] == 1


def test_sequence_classifier_backend_returns_two_choice_logits():
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "attention_mask": torch.ones(4, 2, dtype=torch.long),
        "sample_ids": ["q1", "q2"],
    }
    logits = choice_logits(
        _SequenceClassifier(),
        batch,
        torch.device("cpu"),
        None,
        "sequence_classification",
        pair_start=1,
        pair_end=2,
    )
    assert logits.shape == (2, 2)
    assert torch.equal(logits[:, 0], torch.tensor([11.0, 15.0]))

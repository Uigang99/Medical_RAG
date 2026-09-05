from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag2_pced_direct_choice.py"
SPEC = importlib.util.spec_from_file_location("pced_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rerank_prior_is_bounded_and_ignores_dense_retrieval_scale() -> None:
    documents = [
        {"retrieval_score": 10.0, "rerank_score": -1.0},
        {"retrieval_score": 20.0, "rerank_score": 2.0},
        {"retrieval_score": 30.0, "rerank_score": 1.0},
    ]
    prior = MODULE.rerank_prior(documents, 1e-4)
    assert prior.shape == (3,)
    assert np.all(prior > 0.0)
    assert np.all(prior < 1.0)
    assert prior[1] > prior[2] > prior[0]
    documents[0]["retrieval_score"] = 1e9
    assert np.allclose(prior, MODULE.rerank_prior(documents, 1e-4))


def test_pced_prior_can_change_winning_expert_without_changing_logits() -> None:
    no = np.zeros(4)
    experts = np.asarray([[3.0, 0.0, 0.0, 0.0], [0.0, 2.9, 0.0, 0.0]])
    choice, expert, _ = MODULE.pced_prediction(
        no, experts, beta=0.0, prior=np.asarray([0.1, 0.9]), gamma=2.5
    )
    assert choice == "B"
    assert expert == 1


def test_semantic_matched_prior_preserves_values_and_changes_order_only() -> None:
    retrieval = np.asarray([0.9, 0.2, 0.6])
    semantic = np.asarray([0.1, 0.8, 0.4])
    matched = MODULE.matched_semantic_prior(retrieval, semantic)
    assert np.allclose(np.sort(matched), np.sort(retrieval))
    assert int(np.argmax(matched)) == int(np.argmax(semantic))


def test_condition_summary_accepts_bounded_single_dataset_cohort() -> None:
    rows = [
        {
            "dataset": "medmcqa",
            "gold_answer": "A",
            "predictions": {"trial": "A"},
        },
        {
            "dataset": "medmcqa",
            "gold_answer": "B",
            "predictions": {"trial": "A"},
        },
    ]
    summary = MODULE.condition_summary(rows, "trial")
    assert summary["medmcqa_accuracy"] == 0.5
    assert summary["medqa_accuracy"] is None
    assert summary["mmlu_pooled_accuracy"] is None
    assert summary["macro8_accuracy"] == 0.5
    assert summary["macro3_accuracy"] == 0.5

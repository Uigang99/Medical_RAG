"""Utilities for source-independent biomedical knowledge-graph extraction."""

from .biomedical_mvp import (
    RelationAssertion,
    aggregate_edge_assertions,
    category_from_tuis,
    infer_relation,
    normalize_surface,
)

__all__ = [
    "RelationAssertion",
    "aggregate_edge_assertions",
    "category_from_tuis",
    "infer_relation",
    "normalize_surface",
]

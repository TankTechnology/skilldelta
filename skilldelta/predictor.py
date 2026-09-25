"""Outcome-independent neighbor selection with target-ID exclusion.

Inputs to predict_gain contain outcomes only for historical support tasks.
The same interface supports both unseen queries and leave-one-task-out audits.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Protocol:
    k: int = 3
    scope: str = "family"
    weighting: str = "nonnegative_cosine"
    score: str = "signed_gain"
    threshold: str = "zero"

    def __post_init__(self):
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1:
            raise ValueError("k must be a positive integer")
        for name, choices in (
            ("scope", {"family", "global"}),
            ("weighting", {"nonnegative_cosine", "uniform"}),
            ("score", {"signed_gain", "positive_label"}),
            ("threshold", {"zero", "support_prevalence"}),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"invalid {name}: {getattr(self, name)}")


def _vectors(values, name):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero embedding")
    return x / norms


def _vector(values, length, name, *, outcome=False):
    x = np.asarray(values, dtype=float)
    if x.shape != (length,) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must have {length} finite entries")
    if outcome and np.any((x < 0) | (x > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return x


def _metadata(values, length, name, *, unique=False):
    x = np.asarray(values, dtype=str)
    if x.shape != (length,):
        raise ValueError(f"{name} must have {length} entries")
    if unique and len(set(x)) != length:
        raise ValueError(f"{name} contains duplicate IDs")
    return x


def neighbor_geometry(query_vectors, support_vectors, query_ids, support_ids,
                      query_families, support_families, protocol):
    """Select support and neighbors without reading any outcomes.

    Similarity ties preserve support input order. A same-ID support task is
    always excluded, including from the prevalence threshold's support pool.
    """
    q = _vectors(query_vectors, "query_vectors")
    x = _vectors(support_vectors, "support_vectors")
    if q.shape[1] != x.shape[1]:
        raise ValueError("query and support embedding dimensions differ")
    qi = _metadata(query_ids, len(q), "query_ids", unique=True)
    si = _metadata(support_ids, len(x), "support_ids", unique=True)
    qf = _metadata(query_families, len(q), "query_families")
    sf = _metadata(support_families, len(x), "support_families")
    similarity = q @ x.T
    geometry = []
    for index in range(len(q)):
        eligible = si != qi[index]
        if protocol.scope == "family":
            eligible &= sf == qf[index]
        pool = np.flatnonzero(eligible)
        neighbors = pool[np.argsort(-similarity[index, pool], kind="mergesort")[:protocol.k]]
        weights = (np.maximum(similarity[index, neighbors], 0.0)
                   if protocol.weighting == "nonnegative_cosine"
                   else np.ones(len(neighbors), dtype=float))
        if len(weights) and weights.sum() == 0:
            weights = np.ones(len(weights), dtype=float)
        geometry.append((pool, neighbors, weights))
    return geometry


def predict_gain(*, query_vectors, support_vectors, query_ids, support_ids,
                 query_families, support_families, skip_outcomes, use_outcomes,
                 protocol=Protocol()):
    """Return scores, thresholds and actions; no query outcome is required.

    signed_gain estimates the increment in success probability; positive_label
    is the historical LogicBench gain-sign score, not an increment estimate.
    Empty support returns score=0 and Skip skill. Strict score > threshold is
    used; equality selects Skip skill.
    """
    geometry = neighbor_geometry(query_vectors, support_vectors, query_ids,
                                 support_ids, query_families, support_families, protocol)
    y0 = _vector(skip_outcomes, len(support_ids), "skip_outcomes", outcome=True)
    y1 = _vector(use_outcomes, len(support_ids), "use_outcomes", outcome=True)
    gains = y1 - y0
    positive = gains > 0
    values = gains if protocol.score == "signed_gain" else positive.astype(float)
    scores = np.zeros(len(geometry), dtype=float)
    thresholds = np.zeros(len(geometry), dtype=float)
    neighbors_out = []
    for index, (pool, neighbors, weights) in enumerate(geometry):
        if len(neighbors):
            scores[index] = (values[neighbors].mean() if protocol.weighting == "uniform"
                             else np.dot(weights, values[neighbors]) / weights.sum())
            if protocol.threshold == "support_prevalence":
                thresholds[index] = positive[pool].mean()
        neighbors_out.append(neighbors.tolist())
    return {"scores": scores, "thresholds": thresholds,
            "use_skill": scores > thresholds, "neighbors": neighbors_out}


def predict_skill_success(*, query_vectors, support_vectors, query_ids,
                          support_ids, query_families, support_families,
                          use_outcomes, protocol=Protocol()):
    """With-skill-only ranking control on exactly the same frozen geometry.

    This function has no input for Skip skill outcomes or paired gain labels.
    It predicts skill-assisted success, evaluated against gain signs later.
    """
    geometry = neighbor_geometry(query_vectors, support_vectors, query_ids,
                                 support_ids, query_families, support_families, protocol)
    y1 = _vector(use_outcomes, len(support_ids), "use_outcomes", outcome=True)
    scores = np.zeros(len(geometry), dtype=float)
    for index, (_, neighbors, weights) in enumerate(geometry):
        if len(neighbors):
            scores[index] = (y1[neighbors].mean() if protocol.weighting == "uniform"
                             else np.dot(weights, y1[neighbors]) / weights.sum())
    return scores

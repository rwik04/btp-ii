import numpy as np

from lfgd.lean_score import (
    compute_lean_scores,
    compute_lean_scores_for_selection,
    normalize_lean_scores,
    variance_gate,
)


def test_normalize_lean_scores_range():
    raw = np.array([-3.0, 0.0, 2.0, 5.0])
    norm = normalize_lean_scores(raw)
    assert np.isclose(norm.min(), -1.0)
    assert np.isclose(norm.max(), 1.0)


def test_compute_lean_scores_for_selection_variance_gate_returns_none():
    embeddings = np.ones((20, 8), dtype=float)
    axis = np.ones(8, dtype=float)
    scores, should_debias = compute_lean_scores_for_selection(embeddings, axis, tau=0.05)
    assert scores is None
    assert should_debias is False


def test_compute_lean_scores_shape():
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(20, 6))
    axis = rng.normal(size=(6,))
    scores = compute_lean_scores(embeddings, axis)
    assert scores.shape == (20,)
    assert isinstance(variance_gate(scores), bool)

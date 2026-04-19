import numpy as np

from lfgd.selection import select_balanced_set, select_balanced_set_exact, select_top_k


def test_select_top_k_returns_highest_relevance_indices():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert select_top_k(scores, 2) == [1, 3]


def test_select_balanced_set_exact_returns_k_indices():
    lean = np.array([-1.0, -0.8, -0.2, 0.2, 0.8, 1.0])
    relevance = np.array([0.9, 0.8, 0.5, 0.5, 0.8, 0.9])
    selected = select_balanced_set_exact(lean, relevance, k=4, alpha=0.5)
    assert selected is not None
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_select_balanced_set_falls_back_or_exact():
    rng = np.random.default_rng(3)
    lean = rng.uniform(-1, 1, size=20)
    relevance = rng.uniform(0, 1, size=20)
    selected = select_balanced_set(lean, relevance, k=6, alpha=0.5)
    assert len(selected) == 6

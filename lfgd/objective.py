"""
Loss functions for LFGD balanced set-selection.

Defines:
- L_utility: complement of average normalised relevance
- L_fairness: Wasserstein-1 distance to Uniform[-1, 1]
- L_combined: weighted combination of utility and fairness
"""

import numpy as np


def wasserstein1_uniform(lean_scores: np.ndarray) -> float:
    """
    Wasserstein-1 distance between empirical lean score distribution
    and Uniform[-1, 1].

    For a set of k lean scores sorted ascending l_(1) ≤ ... ≤ l_(k):
        W₁(P_S, U[-1,1]) = (1/k) * Σ |l_(j) - (-1 + (2j-1)/k)|

    Args:
        lean_scores: Array of k lean scores (typically in [-1, 1])

    Returns:
        Wasserstein-1 distance (lower = more uniform/balanced)
    """
    k = len(lean_scores)
    sorted_l = np.sort(lean_scores)
    # Quantiles of Uniform[-1, 1] at midpoints of k equal intervals
    target = np.array([-1.0 + (2.0 * j - 1.0) / k for j in range(1, k + 1)])
    return float(np.mean(np.abs(sorted_l - target)))


def L_utility(selected_indices: list[int], relevance_scores: np.ndarray) -> float:
    """
    Utility loss: complement of average normalised relevance of selected set.

    L_utility(S) = 1 - (1/k) * Σ (r_i / max_j(r_j)) for i ∈ S

    Args:
        selected_indices: List of k indices into relevance_scores
        relevance_scores: Array of shape (N,) with relevance scores r_i

    Returns:
        Utility loss in [0, 1] (lower = better utility)
    """
    if not selected_indices:
        return 1.0

    k = len(selected_indices)
    r_selected = relevance_scores[selected_indices]
    max_r = relevance_scores.max()

    if max_r == 0:
        return 1.0

    avg_normalised = np.mean(r_selected / max_r)
    return float(1.0 - avg_normalised)


def L_fairness(selected_indices: list[int], lean_scores: np.ndarray) -> float:
    """
    Fairness loss: Wasserstein-1 distance to uniform distribution.

    Args:
        selected_indices: List of k indices into lean_scores
        lean_scores: Array of shape (N,) with lean scores l_i

    Returns:
        Fairness loss in [0, 1] (lower = more balanced)
    """
    if not selected_indices:
        return 1.0

    l_selected = lean_scores[selected_indices]
    return wasserstein1_uniform(l_selected)


def L_combined(
    selected_indices: list[int],
    lean_scores: np.ndarray,
    relevance_scores: np.ndarray,
    alpha: float = 0.5,
) -> float:
    """
    Combined loss: α·L_utility + (1-α)·L_fairness.

    Args:
        selected_indices: List of k indices
        lean_scores: Array of shape (N,) with lean scores
        relevance_scores: Array of shape (N,) with relevance scores
        alpha: Trade-off parameter in [0, 1]
               α=1 → utility only, α=0 → fairness only

    Returns:
        Combined loss (lower = better)
    """
    l_util = L_utility(selected_indices, relevance_scores)
    l_fair = L_fairness(selected_indices, lean_scores)
    return float(alpha * l_util + (1.0 - alpha) * l_fair)
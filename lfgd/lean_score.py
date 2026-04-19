"""
Lean score computation for LFGD.

Projects each document embedding onto the bias axis to obtain a
continuous lean score, then normalizes to [-1, 1] and applies
a variance gate to detect ideologically homogeneous candidate sets.
"""

import numpy as np


def compute_lean_scores(
    embeddings: np.ndarray,
    bias_axis: np.ndarray,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute lean scores by projecting embeddings onto the bias axis.

    l_i = u1^T @ (v_i - mu)

    Args:
        embeddings: Array of shape (N, d) with document embeddings
        bias_axis: Bias axis vector of shape (d,)
        mu: Optional mean vector. If None, computes from embeddings.

    Returns:
        Lean scores array of shape (N,)
    """
    if mu is None:
        mu = embeddings.mean(axis=0)

    # Mean-center and project
    V_c = embeddings - mu
    lean_scores = V_c @ bias_axis  # shape (N,)

    return lean_scores


def normalize_lean_scores(lean_scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize lean scores to [-1, 1].

    l_norm = (l - l.min()) / (l.max() - l.min()) * 2 - 1

    Args:
        lean_scores: Raw lean scores array of shape (N,)

    Returns:
        Normalized lean scores in [-1, 1]

    Raises:
        ValueError: If lean_scores has zero range (all same values)
    """
    l_min = lean_scores.min()
    l_max = lean_scores.max()
    l_range = l_max - l_min

    if l_range == 0:
        raise ValueError("All lean scores are identical - cannot normalize")

    l_norm = (lean_scores - l_min) / l_range * 2.0 - 1.0
    return l_norm


def variance_gate(normalized_lean_scores: np.ndarray, tau: float = 0.05) -> bool:
    """
    Check if lean score variance exceeds threshold.

    If variance is too low, the candidate set is deemed ideologically
    homogeneous and debiasing should be skipped.

    Args:
        normalized_lean_scores: Lean scores in [-1, 1]
        tau: Variance threshold (default 0.05)

    Returns:
        True if debiasing should proceed (variance >= tau)
        False if should fall back to top-k (variance < tau)
    """
    return float(np.var(normalized_lean_scores)) >= tau


def compute_lean_scores_for_selection(
    embeddings: np.ndarray,
    bias_axis: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray, bool] | tuple[None, bool]:
    """
    Full lean score computation pipeline for set selection.

    1. Project embeddings onto bias axis
    2. Normalize to [-1, 1]
    3. Check variance gate

    Args:
        embeddings: Array of shape (N, d) with document embeddings
        bias_axis: Bias axis vector of shape (d,)
        tau: Variance threshold (default 0.05)

    Returns:
        Tuple of (normalized_lean_scores, should_debias) where:
        - normalized_lean_scores: Lean scores in [-1, 1], or None if variance gate fails
        - should_debias: True if variance gate passed, False to fall back to top-k
    """
    mu = embeddings.mean(axis=0)
    raw_scores = compute_lean_scores(embeddings, bias_axis, mu)

    try:
        normalized = normalize_lean_scores(raw_scores)
    except ValueError:
        # All identical scores - skip debiasing
        return None, False

    should_debias = variance_gate(normalized, tau)
    if not should_debias:
        return None, False
    return normalized, True

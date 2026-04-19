"""
PCA-based bias axis estimation for LFGD.

Uses thin SVD (dual form) to extract the dominant direction of variance
from the mean-centered embedding matrix. Avoids computing d×d covariance matrix.
"""

import numpy as np


def estimate_bias_axis(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the bias axis from a candidate embedding matrix using PCA.

    Uses thin SVD on the mean-centered embedding matrix to find the
    first principal component (dominant direction of variance).

    Args:
        embeddings: Array of shape (N, d) where N is candidate count,
                   d is embedding dimensionality

    Returns:
        Tuple of (bias_axis, mean, centered_embeddings) where:
        - bias_axis: First right singular vector of shape (d,)
        - mean: Mean vector of embeddings of shape (d,)
        - centered_embeddings: Mean-centered embeddings of shape (N, d)

    Raises:
        ValueError: If embeddings has wrong shape or N < 2
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array (N, d), got {embeddings.ndim}D")

    N, d = embeddings.shape
    if N < 2:
        raise ValueError(f"Need at least 2 candidates, got {N}")

    # Mean-center the embedding matrix
    mu = embeddings.mean(axis=0)  # shape (d,)
    V_c = embeddings - mu  # shape (N, d)

    # Thin SVD: V_c = U @ S @ Vt
    # Vt has shape (d, N), so Vt[0] is the first right singular vector
    _, _, Vt = np.linalg.svd(V_c, full_matrices=False)

    u1 = Vt[0]  # first right singular vector, shape (d,)

    return u1, mu, V_c


def resolve_axis_sign(
    embeddings: np.ndarray, u1: np.ndarray, V_c: np.ndarray
) -> np.ndarray:
    """
    Resolve the sign ambiguity of the bias axis.

    Chooses the sign such that documents from the first half of the
    candidate list project to positive lean scores on average.

    This heuristic exploits the fact that dense retrievers tend to
    surface slightly more of whatever ideological slant the corpus
    leans toward in the top positions.

    Args:
        embeddings: Original embeddings array (N, d)
        u1: First right singular vector from SVD (d,)
        V_c: Mean-centered embeddings (N, d)

    Returns:
        Bias axis with resolved sign (d,)
    """
    N = embeddings.shape[0]
    half = N // 2

    # Mean projection of first half onto u1
    first_half_mean = V_c[:half].mean(axis=0)
    projection = np.dot(first_half_mean, u1)

    if projection < 0:
        u1 = -u1

    return u1


def estimate_bias_axis_with_sign(embeddings: np.ndarray) -> np.ndarray:
    """
    Convenience function: estimate bias axis and resolve sign in one call.

    Args:
        embeddings: Array of shape (N, d) with candidate embeddings

    Returns:
        Bias axis vector of shape (d,) with resolved sign

    Raises:
        ValueError: If embeddings has wrong shape
    """
    u1, _, V_c = estimate_bias_axis(embeddings)
    u1 = resolve_axis_sign(embeddings, u1, V_c)
    return u1
"""
Lean variance diagnostic metric.

Measures the spread of lean scores in a selected set.
Higher variance indicates better coverage of the ideological spectrum.
"""

import numpy as np


def lean_variance(lean_scores: list[float] | np.ndarray) -> float:
    """
    Compute variance of lean scores.

    This is a diagnostic metric, not an objective.
    Higher variance indicates better ideological coverage.

    Args:
        lean_scores: List or array of lean scores (should be in [-1, 1])

    Returns:
        Variance of lean scores
    """
    arr = np.asarray(lean_scores)
    return float(np.var(arr))


def lean_score_stats(lean_scores: list[float] | np.ndarray) -> dict:
    """
    Compute statistics on lean scores.

    Args:
        lean_scores: List or array of lean scores

    Returns:
        Dictionary with min, max, mean, var, range
    """
    arr = np.asarray(lean_scores)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "variance": float(np.var(arr)),
        "range": float(arr.max() - arr.min()),
    }


def balance_ratio(lean_scores: list[float] | np.ndarray) -> float:
    """
    Compute a simple balance ratio: fraction of docs on each side of 0.

    For perfectly balanced selection, this should be close to 0.5.

    Args:
        lean_scores: List or array of lean scores

    Returns:
        Fraction of documents with lean_score < 0
    """
    arr = np.asarray(lean_scores)
    return float(np.mean(arr < 0))
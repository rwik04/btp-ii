"""
C-FAIR: Continuous-FAIR metric for label-free evaluation.

Extends the discrete FAIR metric to continuous lean score distributions
using Wasserstein-1 distance instead of KL divergence.
"""

import math
from typing import List

import numpy as np

from lfgd.objective import wasserstein1_uniform


def compute_cfair_score(
    lean_scores_assembled: List[float],
    relevance_scores: List[float],
    k: int | None = None,
) -> float:
    """
    Compute C-FAIR score for an assembled context.

    C-FAIR = (1/M) * Σ_{i=1}^{k} [g_i / log₂(i+1)] * [1 / (W₁(P_i, U[-1,1]) + 1)]

    where:
      - g_i is the cosine similarity (relevance gain) of document i
      - P_i is the empirical distribution of lean scores of the first i documents
      - W₁(P_i, U[-1,1]) is the Wasserstein-1 distance to uniform
      - M is the normalisation constant

    Args:
        lean_scores_assembled: Lean scores of k documents in assembled order
        relevance_scores: Relevance scores (cosine similarities) for each document
        k: Optional truncation point. Defaults to len(lean_scores_assembled).

    Returns:
        C-FAIR score in [0, 1] (higher = fairer and more relevant)

    Raises:
        ValueError: If lengths of lean_scores and relevance_scores don't match
    """
    if len(lean_scores_assembled) != len(relevance_scores):
        raise ValueError(
            f"Length mismatch: {len(lean_scores_assembled)} lean scores "
            f"vs {len(relevance_scores)} relevance scores"
        )

    if not lean_scores_assembled:
        return 0.0

    if k is None:
        k = len(lean_scores_assembled)
    k = min(k, len(lean_scores_assembled))

    cumulative_score = 0.0

    for i in range(1, k + 1):
        # Prefix of lean scores up to position i
        prefix_leans = np.array(lean_scores_assembled[:i])

        # Wasserstein-1 distance of prefix to Uniform[-1, 1]
        w1 = wasserstein1_uniform(prefix_leans)
        fairness_component = 1.0 / (w1 + 1.0)

        # Relevance gain at position i
        g_i = relevance_scores[i - 1]  # 0-indexed

        # Position discount
        position_discount = 1.0 / math.log2(i + 1)

        cumulative_score += (g_i * position_discount) * fairness_component

    # Compute normalisation constant M (ideal case: perfect relevance + perfect balance)
    ideal_score = 0.0
    ideal_leans = np.array([-1.0 + (2.0 * j - 1.0) / k for j in range(1, k + 1)])
    for i in range(1, k + 1):
        prefix_ideal = ideal_leans[:i]
        w1_ideal = wasserstein1_uniform(prefix_ideal)
        ideal_score += (1.0 * 1.0 / math.log2(i + 1)) * (1.0 / (w1_ideal + 1.0))

    if ideal_score == 0:
        return 0.0

    return float(cumulative_score / ideal_score)


def compute_cfair_at_k(
    lean_scores_assembled: List[float],
    relevance_scores: List[float],
    k: int,
) -> float:
    """
    Compute C-FAIR at position k (prefix evaluation).

    Args:
        lean_scores_assembled: Lean scores in assembled order
        relevance_scores: Relevance scores
        k: Evaluation depth

    Returns:
        C-FAIR@k score
    """
    return compute_cfair_score(lean_scores_assembled, relevance_scores, k=k)
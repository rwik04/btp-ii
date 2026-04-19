"""
ReFaRAG baseline: Probabilistic single-document reranking.

ReFaRAG assigns each document a probability of being selected based on
its relevance score, then samples to build a balanced set. This is a
simpler approach than LFGD's exact combinatorial search.

Reference: https://arxiv.org/abs/2404.07963
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ReFaRAGResult:
    """Result from ReFaRAG reranking."""

    selected_indices: List[int]
    selected_probs: List[float]


def compute_selection_probabilities(
    relevance_scores: List[float],
    temperature: float = 1.0,
) -> List[float]:
    """
    Compute softmax probabilities over relevance scores.

    Args:
        relevance_scores: List of relevance scores
        temperature: Softmax temperature (higher = more uniform)

    Returns:
        List of selection probabilities
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scores = np.array(relevance_scores, dtype=np.float64)
    if scores.size == 0:
        return []

    # Replace non-finite scores with worst finite value (or zero if none finite).
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        scores = np.zeros_like(scores)
    else:
        min_finite = np.min(scores[finite_mask])
        scores = np.where(finite_mask, scores, min_finite)

    scores = scores / temperature

    # Numerically stable softmax.
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    total = exp_scores.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.full(scores.shape, 1.0 / len(scores), dtype=np.float64)
    else:
        probs = exp_scores / total

    # Ensure valid simplex for np.random.choice.
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.full(scores.shape, 1.0 / len(scores), dtype=np.float64)
    else:
        probs = probs / total
    return list(probs)


def sample_balanced_set(
    relevance_scores: List[float],
    lean_scores: List[float],
    k: int = 6,
    num_samples: int = 100,
    seed: int | None = None,
) -> tuple[List[int], List[float]]:
    """
    Sample multiple sets and pick the most balanced one.

    For each sample, selects documents with probability proportional to
    their relevance score until k documents are chosen. Then picks
    the sample whose selected lean score distribution is closest to uniform.

    Args:
        relevance_scores: List of relevance scores
        lean_scores: List of lean scores in [-1, 1]
        k: Number of documents to select
        num_samples: Number of candidate sets to try
        seed: Random seed

    Returns:
        Tuple of (best_indices, probabilities)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    probs = compute_selection_probabilities(relevance_scores)
    N = len(relevance_scores)
    k = min(k, N)
    if k == 0:
        return [], []

    best_indices: List[int] = []
    best_balance_score = float("inf")

    for _ in range(num_samples):
        # Sample k documents with replacement (ReFaRAG uses sampling with replacement)
        p = np.array(probs, dtype=np.float64)
        p = np.clip(p, 0.0, None)
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            p = np.full((N,), 1.0 / N, dtype=np.float64)
        else:
            p = p / p_sum

        sampled = list(np.random.choice(N, size=k, replace=False, p=p))
        sampled_leans = [lean_scores[i] for i in sampled]

        # Compute balance score: distance from uniform distribution
        sorted_leans = sorted(sampled_leans)
        ideal_leans = [-1 + (2 * j - 1) / k for j in range(1, k + 1)]
        balance_score = sum(abs(l - i) for l, i in zip(sorted_leans, ideal_leans))

        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_indices = sampled

    # Get probabilities for selected docs
    selected_probs = [probs[i] for i in best_indices]

    return best_indices, selected_probs


def refarag_rerank(
    relevance_scores: List[float],
    lean_scores: List[float],
    candidate_texts: List[str] | None = None,
    k: int = 6,
    num_samples: int = 100,
    seed: int | None = None,
) -> ReFaRAGResult:
    """
    ReFaRAG probabilistic reranking for balanced retrieval.

    Args:
        relevance_scores: List of relevance scores
        lean_scores: List of lean scores (for balance evaluation)
        candidate_texts: Optional list of candidate texts (unused, for API compat)
        k: Number of documents to select
        num_samples: Number of candidate sets to sample
        seed: Random seed

    Returns:
        ReFaRAGResult with selected indices and probabilities
    """
    indices, probs = sample_balanced_set(
        relevance_scores, lean_scores, k=k, num_samples=num_samples, seed=seed
    )

    return ReFaRAGResult(
        selected_indices=indices,
        selected_probs=probs,
    )

"""
Original FAIR score metric from FairRAG-Select (BTP-I).

This requires discrete group labels [l]/[r] and is used only for
evaluation purposes - it is NOT used by LFGD at inference time.
"""

import math
from typing import Dict, List, Literal

import numpy as np


Group = Literal["l", "r"]


def calculate_kl_divergence(
    current_counts: Dict[Group, int],
    ideal_prob: float = 0.5,
) -> float:
    """
    KL divergence d_KL(D_r || D*) where:
      - D_r is the empirical distribution over groups ('l', 'r') in top-i
      - D* is the ideal distribution (default 50/50)
    """
    total = current_counts.get("l", 0) + current_counts.get("r", 0)
    if total == 0:
        return 0.0

    kl_sum = 0.0
    for group in ("l", "r"):
        p_r = current_counts.get(group, 0) / total
        p_star = ideal_prob
        if p_r == 0:
            p_r = 1e-12
        kl_sum += p_r * np.log(p_r / p_star)
    return float(kl_sum)


def calculate_fair_metric_at_k(
    retrieved_groups_in_rank_order: List[Group],
    k: int,
    ideal_prob: float = 0.5,
    relevance_gain: float = 1.0,
) -> float:
    """
    FAIR = (1/M) * sum_{i=1..k} IRM_i / (d_KL(D_r^i || D*) + 1)

    where:
      - IRM_i = relevance_gain * (1/log2(i+1)) is the position-discounted gain
      - D_r^i is the cumulative group distribution up to rank i
      - D* is the ideal distribution (default 50/50)
      - M is the ideal score (all d_KL = 0)

    Args:
        retrieved_groups_in_rank_order: List of 'l'/'r' labels in rank order
        k: Position up to which to evaluate
        ideal_prob: Ideal proportion for each group (default 0.5 for 50/50)
        relevance_gain: Scaling factor for relevance (default 1.0)

    Returns:
        FAIR score in [0, 1] (higher = fairer)
    """
    if not retrieved_groups_in_rank_order:
        return 0.0

    k = min(k, len(retrieved_groups_in_rank_order))
    cumulative_score = 0.0
    current_counts: Dict[Group, int] = {"l": 0, "r": 0}

    for i in range(k):
        rank = i + 1
        group = retrieved_groups_in_rank_order[i]
        current_counts[group] = current_counts.get(group, 0) + 1

        d_kl = calculate_kl_divergence(current_counts, ideal_prob=ideal_prob)
        fairness_component = 1.0 / (d_kl + 1.0)
        position_discount = 1.0 / math.log2(rank + 1)
        utility_component = relevance_gain * position_discount
        cumulative_score += utility_component * fairness_component

    # Ideal score: perfect fairness (d_KL=0) and perfect utility
    ideal_score = 0.0
    for i in range(k):
        rank = i + 1
        position_discount = 1.0 / math.log2(rank + 1)
        ideal_score += relevance_gain * position_discount * 1.0  # (1 / (0 + 1))

    if ideal_score == 0:
        return 0.0
    return float(cumulative_score / ideal_score)
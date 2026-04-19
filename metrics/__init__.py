"""
Metrics for evaluating LFGD and baseline systems.

- fair: Original FAIR score (requires labels, for eval only)
- cfair: Continuous C-FAIR score (label-free)
- lean_variance: Diagnostic metrics for lean score distributions
"""

from metrics.fair import calculate_fair_metric_at_k, calculate_kl_divergence
from metrics.cfair import compute_cfair_score, compute_cfair_at_k
from metrics.lean_variance import lean_variance, lean_score_stats, balance_ratio

__all__ = [
    "calculate_fair_metric_at_k",
    "calculate_kl_divergence",
    "compute_cfair_score",
    "compute_cfair_at_k",
    "lean_variance",
    "lean_score_stats",
    "balance_ratio",
]

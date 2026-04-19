"""
Analysis modules for LFGD results.

- axis_correlation: H3 correlation between PCA axis and label-derived axis
- results_summary: Aggregate neutrality rates, FAIR, C-FAIR tables
"""

from analysis.axis_correlation import (
    compute_axis_correlation,
    compute_label_axis,
    analyze_axis_correlation,
)
from analysis.results_summary import summarize_results, print_summary

__all__ = [
    "compute_axis_correlation",
    "compute_label_axis",
    "analyze_axis_correlation",
    "summarize_results",
    "print_summary",
]

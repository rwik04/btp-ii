"""
LFGD: Label-Free Geometric Debiasing for RAG.

Core modules:
- pca_axis: Bias axis estimation via PCA on candidate embeddings
- lean_score: Lean score computation and normalization
- selection: Balanced set-selection via constrained optimization
- objective: Loss functions (utility, fairness, combined)
- assemble: Context assembly with lean-score interleaving
"""

from lfgd.pca_axis import estimate_bias_axis, estimate_bias_axis_with_sign, resolve_axis_sign
from lfgd.lean_score import (
    compute_lean_scores,
    normalize_lean_scores,
    variance_gate,
    compute_lean_scores_for_selection,
)
from lfgd.selection import select_balanced_set, select_top_k
from lfgd.objective import L_utility, L_fairness, L_combined, wasserstein1_uniform
from lfgd.assemble import (
    ScoredDocument,
    interleave_by_lean,
    assemble_context,
    format_document_for_synthesis,
)

__all__ = [
    # PCA axis
    "estimate_bias_axis",
    "estimate_bias_axis_with_sign",
    "resolve_axis_sign",
    # Lean scores
    "compute_lean_scores",
    "normalize_lean_scores",
    "variance_gate",
    "compute_lean_scores_for_selection",
    # Selection
    "select_balanced_set",
    "select_top_k",
    # Objectives
    "L_utility",
    "L_fairness",
    "L_combined",
    "wasserstein1_uniform",
    # Assembly
    "ScoredDocument",
    "interleave_by_lean",
    "assemble_context",
    "format_document_for_synthesis",
]

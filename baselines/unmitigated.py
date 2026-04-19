"""
Unmitigated baseline: plain top-k retrieval by relevance.

This baseline simply selects the top-k most relevant documents
without any fairness intervention. Used as a comparison point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lfgd.assemble import ScoredDocument


@dataclass
class UnmitigatedResult:
    """Result from unmitigated retrieval."""

    selected_indices: List[int]
    selected_docs: List[ScoredDocument]


def select_unmitigated(
    relevance_scores: List[float],
    candidate_texts: List[str],
    candidate_embeddings: List | None = None,
    k: int = 6,
) -> UnmitigatedResult:
    """
    Select top-k most relevant documents with no fairness intervention.

    Simply ranks by relevance score and takes the top k.

    Args:
        relevance_scores: List of relevance scores (cosine similarities)
        candidate_texts: List of candidate document texts
        candidate_embeddings: Optional list of embeddings (unused in this baseline)
        k: Number of documents to select

    Returns:
        UnmitigatedResult with selected indices and ScoredDocument objects
    """
    if len(relevance_scores) != len(candidate_texts):
        raise ValueError(
            f"Length mismatch: {len(relevance_scores)} scores vs {len(candidate_texts)} texts"
        )

    # Rank by relevance descending
    indexed_scores = list(enumerate(relevance_scores))
    ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    # Take top-k
    top_k = ranked[:k]
    selected_indices = [idx for idx, _ in top_k]

    # Build ScoredDocument objects
    selected_docs = []
    for idx in selected_indices:
        selected_docs.append(
            ScoredDocument(
                text=candidate_texts[idx],
                lean_score=0.0,  # No lean score (not computed)
                relevance_score=relevance_scores[idx],
                metadata={"index": idx},
            )
        )

    return UnmitigatedResult(
        selected_indices=selected_indices,
        selected_docs=selected_docs,
    )

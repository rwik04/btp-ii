"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines BM25 and dense retrieval using RRF to produce a unified ranking.
The fusion score is: RRF(score) = 1 / (k + rank) where rank is the position
in eachretriever's ranked list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from retrieval.bm25 import BM25Result, BM25Retriever
from retrieval.dense import DenseResult, DenseRetriever


@dataclass
class HybridResult:
    """A hybrid retrieval result combining BM25 and dense scores."""

    index: int
    fused_score: float
    bm25_score: float
    dense_score: float
    text: str
    metadata: dict | None = None


def reciprocal_rank_fusion(
    results_a: List[tuple[int, float]],
    results_b: List[tuple[int, float]],
    k: int = 60,
) -> dict[int, float]:
    """
    Fuse two ranked result lists using Reciprocal Rank Fusion.

    RRF(score) = 1 / (k + rank) where rank is 1-indexed position in list.
    Higher final scores = better.

    Args:
        results_a: List of (index, score) from retriever A, in descending score order
        results_b: List of (index, score) from retriever B, in descending score order
        k: RRF parameter (default 60, higher = more weight to lower ranks)

    Returns:
        Dict mapping document index to fused RRF score
    """
    fused: dict[int, float] = {}

    # Add scores from first retriever
    for rank, (idx, score) in enumerate(results_a, start=1):
        fused[idx] = fused.get(idx, 0) + 1 / (k + rank)

    # Add scores from second retriever
    for rank, (idx, score) in enumerate(results_b, start=1):
        fused[idx] = fused.get(idx, 0) + 1 / (k + rank)

    return fused


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense retrieval via RRF.

    Maintains BM25 and dense retrievers internally and fuses results
    using reciprocal rank fusion.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_retriever: Initialized BM25 retriever
            dense_retriever: Initialized dense retriever
            rrf_k: RRF parameter (default 60)
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[HybridResult]:
        """
        Search using hybrid fusion of BM25 and dense results.

        Args:
            query_text: Raw query text (for BM25)
            query_embedding: Query embedding (for dense)
            k: Number of top results to return

        Returns:
            List of HybridResult sorted by fused_score descending
        """
        import numpy as np

        # Get BM25 results
        bm25_results = self.bm25.search(query_text, k=k * 2)
        bm25_ranked = [(r.index, r.score) for r in bm25_results]

        # Get dense results
        dense_results = self.dense.search(query_embedding, k=k * 2)
        dense_ranked = [(r.index, r.score) for r in dense_results]

        # Fuse via RRF
        fused_scores = reciprocal_rank_fusion(bm25_ranked, dense_ranked, k=self.rrf_k)

        # Get top-k fused results
        top_k_idx = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)[:k]

        # Build result objects with component scores
        index_to_bm25 = {r.index: r.score for r in bm25_results}
        index_to_dense = {r.index: r.score for r in dense_results}

        results = []
        for idx in top_k_idx:
            results.append(
                HybridResult(
                    index=idx,
                    fused_score=fused_scores[idx],
                    bm25_score=index_to_bm25.get(idx, 0.0),
                    dense_score=index_to_dense.get(idx, 0.0),
                    text=self.dense.texts[idx],
                    metadata=None,
                )
            )
        return results

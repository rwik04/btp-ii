"""
Retrieval modules for hybrid BM25 + dense retrieval.

- bm25: BM25Okapi keyword retrieval
- dense: Sentence-transformers embedding + cosine similarity
- hybrid: Reciprocal Rank Fusion of BM25 and dense scores
"""

from retrieval.bm25 import BM25Retriever, BM25Result, create_bm25_retriever
from retrieval.dense import (
    DenseRetriever,
    DenseResult,
    QdrantDenseRetriever,
    cosine_similarity,
    batch_cosine_similarities,
)
from retrieval.hybrid import HybridRetriever, HybridResult, reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "BM25Result",
    "create_bm25_retriever",
    "DenseRetriever",
    "DenseResult",
    "QdrantDenseRetriever",
    "cosine_similarity",
    "batch_cosine_similarities",
    "HybridRetriever",
    "HybridResult",
    "reciprocal_rank_fusion",
]

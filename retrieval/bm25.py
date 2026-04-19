"""
BM25 retrieval using rank_bm25.

Provides fast keyword-based retrieval over tokenized document chunks.
Used in hybrid retrieval alongside dense embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    """A BM25 retrieval result with score."""

    index: int
    score: float
    text: str
    metadata: dict | None = None


class BM25Retriever:
    """
    BM25 retriever for keyword-based document retrieval.

    Tokenizes documents and builds a BM25 index for fast retrieval.
    Can be queried with raw text strings.
    """

    def __init__(self, tokenize_lowercase: bool = True):
        """
        Initialize BM25 retriever.

        Args:
            tokenize_lowercase: Whether to lowercase tokens (default True)
        """
        self.tokenize_lowercase = tokenize_lowercase
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def index(self, texts: List[str], metadata: List[dict | None] | None = None) -> None:
        """
        Build BM25 index from documents.

        Args:
            texts: List of document text strings
            metadata: Optional list of metadata dicts (one per document)
        """
        self.documents = texts
        self.tokenized_docs = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer with optional lowercasing."""
        tokens = text.lower().split() if self.tokenize_lowercase else text.split()
        return tokens

    def search(self, query: str, k: int = 10) -> List[BM25Result]:
        """
        Search BM25 index for top-k documents matching the query.

        Args:
            query: Query string
            k: Number of top results to return

        Returns:
            List of BM25Result sorted by score descending
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append(
                BM25Result(
                    index=int(idx),
                    score=float(scores[idx]),
                    text=self.documents[idx],
                    metadata=None,
                )
            )
        return results

    def search_with_scores(self, query: str, k: int = 10) -> tuple[List[int], List[float]]:
        """
        Search BM25 index and return raw indices and scores.

        Args:
            query: Query string
            k: Number of top results to return

        Returns:
            Tuple of (indices, scores) sorted by score descending
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        return list(top_k_idx), list(scores[top_k_idx])


def create_bm25_retriever(
    documents: List[str],
    metadata: List[dict | None] | None = None,
    tokenize_lowercase: bool = True,
) -> BM25Retriever:
    """
    Factory function to create and index a BM25 retriever in one call.

    Args:
        documents: List of document texts
        metadata: Optional metadata per document
        tokenize_lowercase: Whether to lowercase tokens

    Returns:
        Indexed BM25Retriever ready for search
    """
    retriever = BM25Retriever(tokenize_lowercase=tokenize_lowercase)
    retriever.index(documents, metadata)
    return retriever

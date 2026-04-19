"""
Dense embedding retrieval using sentence-transformers.

Provides embedding-based similarity search using cosine similarity.
GPU embedding is offloaded to Modal; this module handles the retrieval logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class DenseResult:
    """A dense retrieval result with embedding and score."""

    index: int
    score: float
    text: str
    embedding: np.ndarray
    metadata: dict | None = None


class DenseRetriever:
    """
    Dense retriever using embedding similarity.

    Uses sentence-transformers for embeddings and computes
    cosine similarity for retrieval. Embeddings can come from
    Modal (GPU) or local CPU computation.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize dense retriever.

        Args:
            embedding_dim: Dimensionality of embeddings (default 384 for MiniLM)
        """
        self.embedding_dim = embedding_dim
        self.texts: List[str] = []
        self.embeddings: np.ndarray | None = None
        self._is_indexed = False

    def index(self, texts: List[str], embeddings: np.ndarray) -> None:
        """
        Index documents with precomputed embeddings.

        Args:
            texts: List of document texts
            embeddings: Array of shape (N, d) with embeddings

        Raises:
            ValueError: If embeddings shape doesn't match texts count or embedding_dim
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"Number of texts ({len(texts)}) must match embeddings rows ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} doesn't match expected {self.embedding_dim}"
            )

        self.texts = texts
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._is_indexed = True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[DenseResult]:
        """
        Search for top-k documents most similar to query embedding.

        Args:
            query_embedding: Query embedding vector of shape (d,)
            k: Number of top results to return

        Returns:
            List of DenseResult sorted by score descending
        """
        if not self._is_indexed or self.embeddings is None:
            raise ValueError("No documents indexed. Call index() first.")

        # Normalize query
        q = query_embedding / np.linalg.norm(query_embedding)
        scores = self.embeddings @ q

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append(
                DenseResult(
                    index=int(idx),
                    score=float(scores[idx]),
                    text=self.texts[idx],
                    embedding=self.embeddings[idx],
                    metadata=None,
                )
            )
        return results

    def search_with_scores(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> tuple[List[int], List[float]]:
        """
        Search and return raw indices and scores.

        Args:
            query_embedding: Query embedding vector
            k: Number of top results

        Returns:
            Tuple of (indices, scores) sorted by score descending
        """
        if not self._is_indexed or self.embeddings is None:
            raise ValueError("No documents indexed. Call index() first.")

        q = query_embedding / np.linalg.norm(query_embedding)
        scores = self.embeddings @ q
        top_k_idx = np.argsort(scores)[::-1][:k]

        return list(top_k_idx), list(scores[top_k_idx])

    @classmethod
    def from_embeddings(
        cls,
        texts: List[str],
        embeddings: np.ndarray,
    ) -> "DenseRetriever":
        """
        Factory to create retriever and index in one call.

        Args:
            texts: List of document texts
            embeddings: Array of shape (N, d)

        Returns:
            Indexed DenseRetriever
        """
        retriever = cls(embedding_dim=embeddings.shape[1])
        retriever.index(texts, embeddings)
        return retriever


class QdrantDenseRetriever:
    """Dense retriever backed by Qdrant vector search."""

    def __init__(
        self,
        embedding_dim: int = 384,
        collection_name: str | None = None,
        recreate_on_index: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "twinviews-13k")
        self.recreate_on_index = recreate_on_index
        self.texts: List[str] = []
        self.embeddings: np.ndarray | None = None
        self._is_indexed = False
        self._client = self._build_client()

    def collection_exists(self) -> bool:
        """Return whether the configured collection already exists."""
        return self._client.collection_exists(self.collection_name)

    def attach_corpus_cache(self, texts: List[str], embeddings: np.ndarray) -> None:
        """
        Attach local corpus arrays without performing any Qdrant upsert.

        Useful when the collection is already indexed and we only need
        metadata/embeddings locally for downstream processing.
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"Number of texts ({len(texts)}) must match embeddings rows ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} doesn't match expected {self.embedding_dim}"
            )
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.texts = texts
        self.embeddings = normalized.astype(np.float32)
        self._is_indexed = True

    def _build_client(self):
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            timeout = float(os.getenv("QDRANT_TIMEOUT_SEC", "120"))
            return QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )

        local_path = Path(os.getenv("QDRANT_PATH", "data/qdrant"))
        local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(local_path))

    def index(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[dict | None] | None = None,
    ) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"Number of texts ({len(texts)}) must match embeddings rows ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} doesn't match expected {self.embedding_dim}"
            )

        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = normalized.astype(np.float32)

        exists = self._client.collection_exists(self.collection_name)
        if self.recreate_on_index and exists:
            self._client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

        batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "256"))
        upsert_timeout = int(os.getenv("QDRANT_UPSERT_TIMEOUT_SEC", "300"))

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_points = []
            for i in range(start, end):
                payload = {"index": i, "text": texts[i]}
                if metadata and metadata[i]:
                    payload["metadata"] = metadata[i]
                batch_points.append(
                    PointStruct(
                        id=i,
                        vector=normalized[i].tolist(),
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
                wait=True,
                timeout=upsert_timeout,
            )

        self.texts = texts
        self.embeddings = normalized
        self._is_indexed = True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[DenseResult]:
        if not self._is_indexed:
            raise ValueError("No documents indexed. Call index() first.")

        q = query_embedding / np.linalg.norm(query_embedding)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=q.tolist(),
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points

        results: List[DenseResult] = []
        for hit in hits:
            payload = hit.payload or {}
            idx = int(payload.get("index", hit.id))
            text = str(payload.get("text", self.texts[idx]))
            embedding = self.embeddings[idx] if self.embeddings is not None else q
            results.append(
                DenseResult(
                    index=idx,
                    score=float(hit.score),
                    text=text,
                    embedding=embedding,
                    metadata=payload.get("metadata"),
                )
            )
        return results

    def search_with_scores(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> tuple[List[int], List[float]]:
        results = self.search(query_embedding, k=k)
        return [r.index for r in results], [r.score for r in results]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: Vector of shape (d,)
        b: Vector of shape (d,)

    Returns:
        Cosine similarity in [-1, 1]
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def batch_cosine_similarities(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between query vectors and target vectors.

    Args:
        queries: Array of shape (M, d)
        targets: Array of shape (N, d)

    Returns:
        Array of shape (M, N) with cosine similarities
    """
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    t_norm = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    return q_norm @ t_norm.T

"""
Preprocessing script for twinviews-13k.

Downloads the dataset, splits documents into chunks,
builds BM25 and Qdrant indexes, and saves artifacts to data/.

Usage:
    uv run python data/preprocess.py
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Try to import modal for GPU embedding, fall back gracefully
_modal_available = False
try:
    import modal
    _modal_available = True
except ImportError:
    modal = None

from data.download import (
    Document,
    PairedTopic,
    create_chunked_documents,
    get_unique_topics,
    load_twinviews_csv,
    load_twinviews_huggingface,
    split_text,
)
from retrieval import QdrantDenseRetriever

load_dotenv()


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Default config
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 20
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_documents(
    paired_docs: List[PairedTopic],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[List[Document], dict]:
    """
    Split paired documents into chunks.

    Args:
        paired_docs: List of PairedTopic objects
        chunk_size: Target words per chunk
        overlap: Overlapping words between chunks

    Returns:
        Tuple of (document list, stats dict)
    """
    documents = create_chunked_documents(paired_docs, chunk_size, overlap)

    stats = {
        "num_paired_docs": len(paired_docs),
        "num_chunks": len(documents),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "l_chunks": sum(1 for d in documents if d.side == "l"),
        "r_chunks": sum(1 for d in documents if d.side == "r"),
    }

    return documents, stats


def build_bm25_index(documents: List[Document]) -> dict:
    """
    Build BM25 index from documents.

    Args:
        documents: List of Document objects

    Returns:
        Dict with BM25 index data (tokenized docs + vocabulary)
    """
    from rank_bm25 import BM25Okapi

    texts = [d.text for d in documents]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    return {
        "bm25": bm25,
        "texts": texts,
        "tokenized": tokenized,
    }


def build_qdrant_index(documents: List[Document], embeddings: np.ndarray) -> dict:
    """Create and populate a Qdrant dense collection for retrieval."""
    d = embeddings.shape[1]
    collection_name = os.getenv("QDRANT_COLLECTION", "twinviews-13k")
    retriever = QdrantDenseRetriever(
        embedding_dim=d,
        collection_name=collection_name,
        recreate_on_index=True,
    )
    texts = [d.text for d in documents]
    metadata = [
        {"side": d.side, "topic": d.topic, "doc_id": d.doc_id}
        for d in documents
    ]
    retriever.index(texts=texts, embeddings=embeddings, metadata=metadata)
    return {
        "collection_name": collection_name,
        "embedding_dim": d,
        "num_documents": len(documents),
    }


def embed_documents_modal(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Embed documents using Modal GPU infrastructure.

    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name

    Returns:
        Array of embeddings (N, d)
    """
    if not _modal_available:
        raise RuntimeError(
            "Modal not available. Install with: uv pip install modal"
        )

    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from modal_app import app, embed_texts

    modal_batch_size = int(os.getenv("MODAL_EMBED_BATCH_SIZE", "512"))
    if modal_batch_size <= 0:
        raise ValueError("MODAL_EMBED_BATCH_SIZE must be > 0")

    # Call Modal function in chunks to avoid one giant RPC payload.
    all_embeddings: list[list[float]] = []
    with app.run():
        for start in range(0, len(texts), modal_batch_size):
            end = min(start + modal_batch_size, len(texts))
            batch_embeddings = embed_texts.remote(texts[start:end], model_name)
            all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def embed_documents_local(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Embed documents using local CPU (sentence-transformers).

    Fallback for when Modal is not used.

    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name

    Returns:
        Array of embeddings (N, d)
    """
    from transformers.utils import logging as hf_logging
    from sentence_transformers import SentenceTransformer

    # Suppress known benign checkpoint key mismatch warnings (e.g. position_ids).
    hf_logging.set_verbosity_error()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def build_indexes(
    documents: List[Document],
    use_modal: bool = False,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> dict:
    """
    Build BM25 and Qdrant indexes for documents.

    Args:
        documents: List of Document objects
        use_modal: Whether to use Modal for GPU embedding
        model_name: Embedding model name
        batch_size: Batch size for embedding

    Returns:
        Dict containing all index artifacts
    """
    print(f"Building indexes for {len(documents)} documents...")

    # Build BM25 index
    print("Building BM25 index...")
    bm25_data = build_bm25_index(documents)
    print("  BM25 index built.")

    # Embed documents
    texts = [d.text for d in documents]
    print(f"Embedding {len(texts)} documents...")

    if use_modal:
        print("  Using Modal GPU for embedding...")
        embeddings = embed_documents_modal(texts, model_name)
    else:
        print("  Using local CPU for embedding...")
        embeddings = embed_documents_local(texts, model_name)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings_normalized = embeddings / norms

    # Build Qdrant index
    print("Building Qdrant index...")
    qdrant_data = build_qdrant_index(documents, embeddings_normalized)
    print(f"  Qdrant collection ready: {qdrant_data['collection_name']}")

    return {
        "bm25_data": bm25_data,
        "qdrant_data": qdrant_data,
        "embeddings": embeddings,
        "embeddings_normalized": embeddings_normalized,
        "documents": documents,
        "model_name": model_name,
    }


def save_indexes(index_data: dict, output_dir: Path = DATA_DIR) -> None:
    """
    Save all index artifacts to disk.

    Args:
        index_data: Dict from build_indexes()
        output_dir: Output directory path
    """
    print(f"Saving indexes to {output_dir}...")

    # Save Qdrant index metadata
    qdrant_meta_path = output_dir / "qdrant_index.json"
    with open(qdrant_meta_path, "w", encoding="utf-8") as f:
        json.dump(index_data["qdrant_data"], f, ensure_ascii=False, indent=2)
    print(f"  Qdrant metadata: {qdrant_meta_path}")

    # Save normalized embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, index_data["embeddings_normalized"])
    print(f"  Embeddings: {embeddings_path}")

    # Save documents metadata (JSON-serializable)
    docs_data = [
        {"text": d.text, "side": d.side, "topic": d.topic, "doc_id": d.doc_id}
        for d in index_data["documents"]
    ]
    docs_path = output_dir / "documents.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, ensure_ascii=False, indent=2)
    print(f"  Documents: {docs_path}")

    # Save BM25 data (tokenized texts + vocab as pickle)
    bm25_path = output_dir / "bm25.pkl"
    bm25_save = {
        "texts": index_data["bm25_data"]["texts"],
        "tokenized": index_data["bm25_data"]["tokenized"],
    }
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_save, f)
    print(f"  BM25 data: {bm25_path}")

    print("All indexes saved.")


def load_indexes(output_dir: Path = DATA_DIR) -> dict:
    """
    Load saved BM25/docs/embeddings artifacts and Qdrant metadata.

    Args:
        output_dir: Directory containing saved indexes

    Returns:
        Dict with all loaded index data
    """
    # Load Qdrant metadata
    qdrant_meta_path = output_dir / "qdrant_index.json"
    if qdrant_meta_path.exists():
        with open(qdrant_meta_path, "r", encoding="utf-8") as f:
            qdrant_data = json.load(f)
    else:
        raise FileNotFoundError(f"Qdrant metadata not found at {qdrant_meta_path}")

    # Load embeddings
    embeddings_path = output_dir / "embeddings.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")

    # Load documents
    docs_path = output_dir / "documents.json"
    if docs_path.exists():
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        documents = [
            Document(text=d["text"], side=d["side"], topic=d["topic"], doc_id=d["doc_id"])
            for d in docs_data
        ]
    else:
        raise FileNotFoundError(f"Documents not found at {docs_path}")

    # Load BM25 data
    bm25_path = output_dir / "bm25.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_save = pickle.load(f)
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(bm25_save["tokenized"])
        bm25_data = {
            "bm25": bm25,
            "texts": bm25_save["texts"],
            "tokenized": bm25_save["tokenized"],
        }
    else:
        raise FileNotFoundError(f"BM25 data not found at {bm25_path}")

    return {
        "qdrant_data": qdrant_data,
        "embeddings": embeddings,
        "documents": documents,
        "bm25_data": bm25_data,
    }


def main():
    """Run full preprocessing pipeline."""
    print("=== BTP-II Data Preprocessing ===")
    print()

    # Load dataset (prefer local CSV from data/download.py if present)
    t0 = time.time()
    local_csv = DATA_DIR / "twinviews-13k.csv"
    if local_csv.exists():
        print(f"Loading twinviews-13k from local CSV: {local_csv}")
        paired_docs = load_twinviews_csv(str(local_csv))
    else:
        print("Local CSV not found. Loading twinviews-13k from HuggingFace...")
        paired_docs = load_twinviews_huggingface()
    print(f"  Loaded {len(paired_docs)} paired documents in {time.time()-t0:.1f}s")
    print(f"  {len(get_unique_topics(paired_docs))} unique topics")

    # Chunk documents
    print("\nChunking documents...")
    t0 = time.time()
    documents, stats = chunk_documents(paired_docs)
    print(f"  Created {stats['num_chunks']} chunks in {time.time()-t0:.1f}s")
    print(f"  Left chunks: {stats['l_chunks']}, Right chunks: {stats['r_chunks']}")

    # Build indexes
    print("\nBuilding indexes...")
    t0 = time.time()

    use_modal = os.getenv("USE_MODAL_EMBEDDING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_modal:
        print("  Will use Modal for GPU embedding")
    else:
        print("  Will use local CPU embedding")

    index_data = build_indexes(documents, use_modal=use_modal)
    print(f"  Indexes built in {time.time()-t0:.1f}s")

    # Save
    print("\nSaving indexes...")
    save_indexes(index_data)

    print("\n=== Preprocessing complete ===")


if __name__ == "__main__":
    main()

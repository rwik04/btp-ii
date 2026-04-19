"""
Data loading and preprocessing for twinviews-13k.

- download: Load dataset from HuggingFace or CSV
- preprocess: Chunk documents, build BM25 + FAISS indexes
"""

from data.download import (
    Document,
    PairedTopic,
    create_chunked_documents,
    get_unique_topics,
    load_twinviews_csv,
    load_twinviews_huggingface,
    split_text,
)

__all__ = [
    "Document",
    "PairedTopic",
    "create_chunked_documents",
    "get_unique_topics",
    "load_twinviews_csv",
    "load_twinviews_huggingface",
    "split_text",
]

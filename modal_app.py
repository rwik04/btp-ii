"""
Modal app configuration for BTP-II LFGD GPU workloads.

This module defines the Modal app with GPU-enabled containers for:
- Embedding generation (sentence-transformers)
- LLM inference (via API - no local GPU needed)

Usage:
    modal deploy modal_app.py
"""

import modal

# Create Modal app
app = modal.App("btp-ii-lfgd")

# GPU image with all ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "datasets>=2.19.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "rank-bm25>=0.2.2",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "openai>=1.30.0",
        "groq>=0.9.0",
        "vaderSentiment>=3.3.2",
        "python-dotenv>=1.0.0",
    )
    .pip_install("torch", extra_options="--index-url https://download.pytorch.org/whl/cu121")
)


@app.function(image=image, gpu="T4")
def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for a list of texts using GPU.

    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name for embeddings

    Returns:
        List of embedding vectors (numpy arrays)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


@app.function(image=image, gpu="T4")
def embed_query(query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embedding for a single query using GPU.

    Args:
        query: Query text to embed
        model_name: HuggingFace model name for embeddings

    Returns:
        Query embedding vector (numpy array)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()


@app.function(image=image, gpu="T4")
def batch_embed(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Batch embedding with progress tracking.

    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name for embeddings

    Returns:
        List of embedding vectors
    """
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    model = SentenceTransformer(model_name)
    embeddings = []
    batch_size = 32

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())

    return embeddings


@app.local_entrypoint()
def main():
    """Test the embedding functions locally."""
    test_texts = ["Hello, world!", "This is a test sentence."]
    print("Testing embed_texts...")
    result = embed_texts.local(test_texts)
    print(f"Got {len(result)} embeddings, each shape: {len(result[0])}")
"""
Axis correlation analysis (H3).

Computes cosine similarity between:
1) PCA-derived axis on retrieved candidates
2) Label-derived axis (mean embedding difference between r and l)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data import load_twinviews_huggingface
from lfgd import estimate_bias_axis_with_sign

_modal_available = False
try:
    import modal

    _modal_available = True
except ImportError:
    modal = None


def _sanitize_model_name(model_name: str) -> str:
    return "".join(c if c.isalnum() or c in {".", "_", "-"} else "_" for c in model_name)


def _hash_text_list(values: List[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def compute_label_axis(embeddings: np.ndarray, labels: List[str]) -> np.ndarray:
    left = embeddings[[i for i, label in enumerate(labels) if label == "l"]]
    right = embeddings[[i for i, label in enumerate(labels) if label == "r"]]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Need at least one l and one r label")
    axis = right.mean(axis=0) - left.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Label-derived axis has zero norm")
    return axis / norm


def compute_axis_correlation(pca_axis: np.ndarray, label_axis: np.ndarray) -> float:
    pca_norm = np.linalg.norm(pca_axis)
    if pca_norm == 0:
        raise ValueError("PCA axis has zero norm")
    return float(np.dot(pca_axis / pca_norm, label_axis))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _batched_topk_indices(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int,
    batch_size: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("n_candidates must be > 0")
    n_docs = doc_embeddings.shape[0]
    k = min(k, n_docs)
    n_queries = query_embeddings.shape[0]

    topk = np.empty((n_queries, k), dtype=np.int32)
    doc_t = doc_embeddings.T

    for start in tqdm(range(0, n_queries, batch_size), desc="Precomputing top-k retrieval"):
        end = min(start + batch_size, n_queries)
        batch = query_embeddings[start:end]
        scores = batch @ doc_t

        kth = max(0, n_docs - k)
        part = np.argpartition(scores, kth=kth, axis=1)[:, -k:]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(part_scores, axis=1)[:, ::-1]
        topk[start:end] = np.take_along_axis(part, order, axis=1)

    return topk


def _encode_with_modal(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    if not _modal_available:
        raise RuntimeError("Modal is not installed. Install with: uv add modal")
    if batch_size <= 0:
        raise ValueError("modal_batch_size must be > 0")

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from modal_app import app, embed_texts

    all_embeddings: list[list[float]] = []
    with app.run():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding on Modal"):
            end = min(start + batch_size, len(texts))
            batch_embeddings = embed_texts.remote(texts[start:end], model_name)
            all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def analyze_axis_correlation(
    results_dir: Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_candidates: int = 20,
    n_topics: int | None = None,
    embedding_cache_dir: Path = Path("data/cache"),
    use_embedding_cache: bool = True,
    retrieval_batch_size: int = 256,
    use_modal_embedding: bool = False,
    modal_batch_size: int = 512,
) -> dict:
    paired_docs = load_twinviews_huggingface()
    if n_topics is not None:
        paired_docs = paired_docs[:n_topics]

    all_texts: list[str] = []
    all_labels: list[str] = []
    for paired in paired_docs:
        all_texts.append(paired.left)
        all_labels.append("l")
        all_texts.append(paired.right)
        all_labels.append("r")

    topics = [p.topic for p in paired_docs]

    model_key = _sanitize_model_name(embedding_model)
    doc_hash = _hash_text_list(all_texts)[:16]
    query_hash = _hash_text_list(topics)[:16]

    doc_cache_path = embedding_cache_dir / f"axis_doc_emb_{model_key}_{doc_hash}.npy"
    query_cache_path = embedding_cache_dir / f"axis_query_emb_{model_key}_{query_hash}.npy"

    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    model: SentenceTransformer | None = None

    if use_embedding_cache and doc_cache_path.exists():
        print(f"Loaded cached document embeddings: {doc_cache_path}")
        embeddings = np.load(doc_cache_path)
    else:
        if use_modal_embedding:
            print(f"Computing document embeddings on Modal for {len(all_texts)} texts...")
            embeddings = _encode_with_modal(
                texts=all_texts,
                model_name=embedding_model,
                batch_size=modal_batch_size,
            )
        else:
            print(f"Computing document embeddings for {len(all_texts)} texts...")
            if model is None:
                model = SentenceTransformer(embedding_model)
            embeddings = model.encode(
                all_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=128,
            )
        if use_embedding_cache:
            np.save(doc_cache_path, embeddings.astype(np.float32))
            print(f"Saved document embedding cache: {doc_cache_path}")

    embeddings = _normalize_rows(embeddings).astype(np.float32)

    if use_embedding_cache and query_cache_path.exists():
        print(f"Loaded cached query embeddings: {query_cache_path}")
        query_embeddings = np.load(query_cache_path)
    else:
        if use_modal_embedding:
            print(f"Computing query embeddings on Modal for {len(topics)} topics...")
            query_embeddings = _encode_with_modal(
                texts=topics,
                model_name=embedding_model,
                batch_size=modal_batch_size,
            )
        else:
            print(f"Computing query embeddings for {len(topics)} topics...")
            if model is None:
                model = SentenceTransformer(embedding_model)
            query_embeddings = model.encode(
                topics,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=128,
            )
        if use_embedding_cache:
            np.save(query_cache_path, query_embeddings.astype(np.float32))
            print(f"Saved query embedding cache: {query_cache_path}")

    query_embeddings = _normalize_rows(query_embeddings).astype(np.float32)

    print(
        f"Computing top-{n_candidates} candidates for {len(query_embeddings)} topics "
        f"with retrieval batch size {retrieval_batch_size}..."
    )
    topk_indices = _batched_topk_indices(
        query_embeddings=query_embeddings,
        doc_embeddings=embeddings,
        k=n_candidates,
        batch_size=retrieval_batch_size,
    )

    per_topic: list[dict] = []
    for i, paired in tqdm(enumerate(paired_docs), total=len(paired_docs), desc="Analyzing topics"):
        candidate_indices = topk_indices[i]
        candidate_embs = embeddings[candidate_indices]
        candidate_labels = [all_labels[j] for j in candidate_indices]

        try:
            pca_axis = estimate_bias_axis_with_sign(candidate_embs)
            label_axis = compute_label_axis(candidate_embs, candidate_labels)
            cosine = compute_axis_correlation(pca_axis, label_axis)
        except ValueError:
            cosine = 0.0

        per_topic.append(
            {
                "topic": paired.topic,
                "cosine": cosine,
                "n_candidates": len(candidate_indices),
            }
        )

    cosines = np.array([x["cosine"] for x in per_topic], dtype=float)
    summary = {
        "embedding_model": embedding_model,
        "n_topics": len(per_topic),
        "n_candidates": n_candidates,
        "mean_cosine": float(np.mean(cosines)) if len(cosines) else 0.0,
        "std_cosine": float(np.std(cosines)) if len(cosines) else 0.0,
        "min_cosine": float(np.min(cosines)) if len(cosines) else 0.0,
        "max_cosine": float(np.max(cosines)) if len(cosines) else 0.0,
        "per_topic": per_topic,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"axis_correlation_{timestamp}.json"
    csv_path = results_dir / f"axis_correlation_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic", "cosine", "n_candidates"])
        writer.writeheader()
        writer.writerows(per_topic)

    summary["json_path"] = str(json_path)
    summary["csv_path"] = str(csv_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PCA/label axis correlation (H3)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--n-topics", type=int, default=None)
    parser.add_argument("--embedding-cache-dir", type=str, default="data/cache")
    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--use-modal-embedding", action="store_true")
    parser.add_argument("--modal-batch-size", type=int, default=512)
    args = parser.parse_args()

    summary = analyze_axis_correlation(
        results_dir=Path(args.results_dir),
        embedding_model=args.embedding_model,
        n_candidates=args.n_candidates,
        n_topics=args.n_topics,
        embedding_cache_dir=Path(args.embedding_cache_dir),
        use_embedding_cache=not args.no_embedding_cache,
        retrieval_batch_size=args.retrieval_batch_size,
        use_modal_embedding=args.use_modal_embedding,
        modal_batch_size=args.modal_batch_size,
    )
    print(f"Mean cosine: {summary['mean_cosine']:.4f}")
    print(f"Saved JSON: {summary['json_path']}")
    print(f"Saved CSV: {summary['csv_path']}")


if __name__ == "__main__":
    main()

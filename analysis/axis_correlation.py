"""
Axis correlation analysis (H3).

Computes cosine similarity between:
1) PCA-derived axis on retrieved candidates
2) Label-derived axis (mean embedding difference between r and l)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_twinviews_huggingface
from lfgd import estimate_bias_axis_with_sign
from retrieval import DenseRetriever


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


def analyze_axis_correlation(
    results_dir: Path,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_candidates: int = 20,
    n_topics: int | None = None,
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

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    retriever = DenseRetriever(embedding_dim=embeddings.shape[1])
    retriever.index(all_texts, embeddings)

    per_topic: list[dict] = []
    for paired in paired_docs:
        query_emb = model.encode(paired.topic, convert_to_numpy=True)
        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0:
            continue
        query_emb = query_emb / q_norm

        candidate_indices, _ = retriever.search_with_scores(query_emb, k=n_candidates)
        candidate_embs = embeddings[candidate_indices]
        candidate_labels = [all_labels[i] for i in candidate_indices]

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
    args = parser.parse_args()

    summary = analyze_axis_correlation(
        results_dir=Path(args.results_dir),
        embedding_model=args.embedding_model,
        n_candidates=args.n_candidates,
        n_topics=args.n_topics,
    )
    print(f"Mean cosine: {summary['mean_cosine']:.4f}")
    print(f"Saved JSON: {summary['json_path']}")
    print(f"Saved CSV: {summary['csv_path']}")


if __name__ == "__main__":
    main()

"""
Main evaluation loop for LFGD and baseline experiments.

Runs the full pipeline for all configured systems on the twinviews-13k
dataset: retrieval -> bias estimation -> selection -> synthesis -> judging.

Usage:
    uv run python experiments/run_eval.py --config experiments/configs/default.yaml
    uv run python experiments/run_eval.py --config experiments/configs/default.yaml --n-topics 30
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Import LFGD core
from lfgd import (
    estimate_bias_axis_with_sign,
    compute_lean_scores_for_selection,
    select_balanced_set,
    ScoredDocument,
)
from lfgd.assemble import assemble_context, interleave_by_lean

# Import metrics
from metrics import (
    calculate_fair_metric_at_k,
    compute_cfair_at_k,
    lean_variance,
)

# Import retrieval
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, QdrantDenseRetriever

# Import generation
from generation import Generator, Judge, JUDGE_CATEGORIES

# Import baselines
from baselines import select_unmitigated, refarag_rerank, fairrag_select

# Import data
from data import load_twinviews_huggingface, get_unique_topics, PairedTopic

load_dotenv()

DEFAULT_N = 20
DEFAULT_K = 6
DEFAULT_ALPHA = 0.5
DEFAULT_TAU = 0.05


@dataclass
class RetrievedDoc:
    """A retrieved document with all metadata needed for selection."""

    text: str
    embedding: np.ndarray
    relevance_score: float
    lean_score: float = 0.0
    side: str = ""  # 'l' or 'r' (only for label-requiring baselines)
    topic: str = ""
    doc_id: str = ""


@dataclass
class EvalResult:
    """Result of evaluating one query."""

    topic: str
    system: str
    question: str
    selected_indices: List[int]
    selected_texts: List[str]
    lean_scores: List[float]
    cfair_score: float
    fair_score: float | None  # Only computed when labels available
    lean_var: float
    synthesis: str
    judge_category: str
    latency_sec: float
    error: str = ""


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _hash_text_list(values: List[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def _load_preprocessed_corpus(
    data_dir: Path = Path("data"),
) -> tuple[list[str], np.ndarray, list[str], list[str]] | None:
    """Load preprocessed retrieval corpus if available."""
    docs_path = data_dir / "documents.json"
    emb_path = data_dir / "embeddings.npy"
    if not docs_path.exists() or not emb_path.exists():
        return None

    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    embeddings = np.load(emb_path)

    texts = [str(d["text"]) for d in docs]
    sides = [str(d.get("side", "")) for d in docs]
    topics = [str(d.get("topic", "")) for d in docs]
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Preprocessed corpus mismatch: {embeddings.shape[0]} embeddings for {len(texts)} documents"
        )
    return texts, embeddings, sides, topics


def build_retriever(
    texts: List[str],
    embeddings: np.ndarray,
    use_hybrid: bool = False,
    reindex_qdrant: bool = False,
) -> HybridRetriever | DenseRetriever | QdrantDenseRetriever:
    """
    Build retrieval indexes.

    Args:
        texts: List of document texts
        embeddings: Array of shape (N, d) with L2-normalized embeddings
        use_hybrid: Whether to use hybrid BM25+dense retrieval
        reindex_qdrant: If True, force upsert/reindex into Qdrant

    Returns:
        HybridRetriever (hybrid mode) or DenseRetriever (dense-only mode)
    """
    dense = QdrantDenseRetriever(
        embedding_dim=embeddings.shape[1],
        recreate_on_index=reindex_qdrant,
    )
    dense.attach_corpus_cache(texts, embeddings)
    if reindex_qdrant:
        dense.index(texts, embeddings)
    elif not dense.collection_exists():
        dense.index(texts, embeddings)

    if use_hybrid:
        bm25 = BM25Retriever()
        bm25.index(texts)
        return HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    return dense


def retrieve_candidates(
    retriever,
    query_text: str,
    query_embedding: np.ndarray,
    N: int = 20,
) -> tuple[List[int], List[float]]:
    """
    Retrieve top-N candidates.

    Args:
        retriever: BM25Retriever, DenseRetriever, or HybridRetriever
        query_text: Query text (for BM25)
        query_embedding: Query embedding (for dense)
        N: Number of candidates

    Returns:
        Tuple of (indices, relevance_scores)
    """
    if isinstance(retriever, HybridRetriever):
        results = retriever.search(query_text, query_embedding, k=N)
        return [r.index for r in results], [r.dense_score for r in results]
    elif isinstance(retriever, (DenseRetriever, QdrantDenseRetriever)):
        indices, scores = retriever.search_with_scores(query_embedding, k=N)
        return indices, scores
    else:
        # BM25
        results = retriever.search(query_text, k=N)
        return [r.index for r in results], [r.score for r in results]


def run_lfgd(
    candidate_docs: List[RetrievedDoc],
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    tau: float = DEFAULT_TAU,
) -> tuple[List[int], List[float]]:
    """
    Run LFGD selection algorithm.

    Args:
        candidate_docs: List of RetrievedDoc with embeddings and relevance scores
        alpha: Fairness weight
        tau: Variance gate threshold

    Returns:
        Tuple of (selected_indices, lean_scores)
    """
    # Stack embeddings
    k = min(k, len(candidate_docs))
    if k == 0:
        return [], []

    embeddings = np.stack([d.embedding for d in candidate_docs])
    relevance_scores = np.array([d.relevance_score for d in candidate_docs])

    # Estimate bias axis
    bias_axis = estimate_bias_axis_with_sign(embeddings)

    # Compute lean scores
    lean_scores, should_debias = compute_lean_scores_for_selection(
        embeddings, bias_axis, tau=tau
    )

    if not should_debias or lean_scores is None:
        # Fallback to top-k by relevance
        indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)[:k]
        return indices, [float(candidate_docs[i].lean_score) for i in indices]

    # Select balanced set
    selected_indices = select_balanced_set(
        lean_scores, relevance_scores, k=k, alpha=alpha
    )

    return selected_indices, lean_scores[selected_indices].tolist()


def run_unmitigated(
    candidate_docs: List[RetrievedDoc],
    k: int = DEFAULT_K,
) -> tuple[List[int], List[float]]:
    """Run unmitigated (top-k by relevance) baseline."""
    k = min(k, len(candidate_docs))
    if k == 0:
        return [], []
    relevance_scores = np.array([d.relevance_score for d in candidate_docs])
    indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)[:k]
    return indices, [float(candidate_docs[i].lean_score) for i in indices]


def run_refarag(
    candidate_docs: List[RetrievedDoc],
    k: int = DEFAULT_K,
) -> tuple[List[int], List[float]]:
    """Run ReFaRAG baseline."""
    k = min(k, len(candidate_docs))
    if k == 0:
        return [], []
    relevance_scores = [d.relevance_score for d in candidate_docs]
    lean_scores = np.array([d.lean_score for d in candidate_docs])

    result = refarag_rerank(relevance_scores, lean_scores, k=k)
    return result.selected_indices, lean_scores[result.selected_indices].tolist()


def run_fairrag_select(
    candidate_docs: List[RetrievedDoc],
    question: str,
    k: int = DEFAULT_K,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[List[int], List[str]]:
    """Run BTP-I FairRAG-Select baseline."""
    k = min(k, len(candidate_docs))
    if k == 0:
        return [], []
    texts = [d.text for d in candidate_docs]
    groups = [d.side for d in candidate_docs]  # Requires explicit labels
    relevance_scores = [d.relevance_score for d in candidate_docs]

    result = fairrag_select(
        candidate_texts=texts,
        candidate_groups=groups,
        relevance_scores=relevance_scores,
        question=question,
        target_k=k,
        model=model,
        provider=provider,
    )

    return result.selected_indices, result.selected_groups


def evaluate_one_query(
    query_idx: int,
    topic: str,
    question: str,
    candidate_docs: List[RetrievedDoc],
    systems: List[str],
    llm_config: dict,
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    tau: float = DEFAULT_TAU,
) -> List[EvalResult]:
    """
    Evaluate all systems on a single query.

    Args:
        query_idx: Index of query for ordering
        topic: Topic name
        question: The question text
        candidate_docs: List of candidate RetrievedDoc objects
        systems: List of system names to evaluate
        llm_config: LLM configuration dict

    Returns:
        List of EvalResult (one per system)
    """
    results = []
    init_errors: list[str] = []
    generator = None
    judge = None
    try:
        generator = Generator(
            model=llm_config.get("synth_model"),
            provider=llm_config.get("synth_provider"),
            reasoning_effort=llm_config.get("synth_reasoning_effort"),
        )
    except Exception as e:
        generator = None
        init_errors.append(f"generator_init: {e}")

    try:
        judge = Judge(
            model=llm_config.get("judge_model"),
            provider=llm_config.get("judge_provider"),
            reasoning_effort=llm_config.get("judge_reasoning_effort"),
        )
    except Exception as e:
        judge = None
        init_errors.append(f"judge_init: {e}")

    for system in systems:
        try:
            start = time.time()
            system_errors = list(init_errors)
            k_effective = min(k, len(candidate_docs))
            if k_effective == 0:
                raise ValueError("No candidate docs to evaluate")

            if system == "lfgd":
                selected_indices, _ = run_lfgd(candidate_docs, k=k_effective, alpha=alpha, tau=tau)

            elif system == "unmitigated":
                selected_indices, _ = run_unmitigated(candidate_docs, k=k_effective)

            elif system == "refarag":
                selected_indices, _ = run_refarag(candidate_docs, k=k_effective)

            elif system == "fairrag_select":
                selected_indices, _ = run_fairrag_select(
                    candidate_docs,
                    question,
                    k=k_effective,
                    model=llm_config.get("fairrag_model"),
                    provider=llm_config.get("fairrag_provider"),
                )

            else:
                continue

            selected_indices = [int(i) for i in selected_indices]
            selected_docs = [candidate_docs[i] for i in selected_indices]
            scored_docs = [
                ScoredDocument(
                    text=d.text,
                    lean_score=float(d.lean_score),
                    relevance_score=float(d.relevance_score),
                    metadata={"side": d.side, "doc_id": d.doc_id, "topic": d.topic},
                )
                for d in selected_docs
            ]
            assembled_docs = interleave_by_lean(scored_docs)
            assembled_leans = [d.lean_score for d in assembled_docs]
            assembled_relevance = [d.relevance_score for d in assembled_docs]

            synthesis_text = assemble_context(assembled_docs)
            if generator is not None:
                try:
                    synthesis_text = generator.synthesize(
                        question,
                        [
                            {
                                "text": d.text,
                                "side": (d.metadata or {}).get("side", "?"),
                                "lean_score": d.lean_score,
                                "relevance_score": d.relevance_score,
                            }
                            for d in assembled_docs
                        ],
                    ).synthesis
                except Exception as e:
                    system_errors.append(f"synthesis: {e}")

            cfair = compute_cfair_at_k(assembled_leans, assembled_relevance, k_effective)
            selected_groups = [
                (d.metadata or {}).get("side", "")
                for d in assembled_docs
                if (d.metadata or {}).get("side", "") in {"l", "r"}
            ]
            fair_score = (
                calculate_fair_metric_at_k(selected_groups, k_effective)
                if len(selected_groups) == len(assembled_docs)
                else None
            )

            judge_category = "Mixed/Unclear"
            if judge is not None:
                try:
                    judge_result = judge.classify(synthesis_text)
                    judge_category = judge_result.category
                except Exception as e:
                    system_errors.append(f"judge: {e}")

            lv = lean_variance(assembled_leans)

            results.append(
                EvalResult(
                    topic=topic,
                    system=system,
                    question=question,
                    selected_indices=selected_indices,
                    selected_texts=[d.text for d in assembled_docs],
                    lean_scores=assembled_leans,
                    cfair_score=cfair,
                    fair_score=fair_score,
                    lean_var=lv,
                    synthesis=synthesis_text,
                    judge_category=judge_category,
                    latency_sec=time.time() - start,
                    error=" | ".join(system_errors),
                )
            )
        except Exception as e:
            results.append(
                EvalResult(
                    topic=topic,
                    system=system,
                    question=question,
                    selected_indices=[],
                    selected_texts=[],
                    lean_scores=[],
                    cfair_score=0.0,
                    fair_score=None,
                    lean_var=0.0,
                    synthesis="",
                    judge_category="",
                    latency_sec=0.0,
                    error=str(e),
                )
            )

    return results


def run_experiment(
    config: dict,
    paired_docs: List[PairedTopic],
) -> List[EvalResult]:
    """
    Run full experiment pipeline.

    Args:
        config: Configuration dict
        paired_docs: List of PairedTopic objects

    Returns:
        List of EvalResult for all queries and systems
    """
    # Load config values
    N = config.get("retrieval", {}).get("N", DEFAULT_N)
    K = config.get("retrieval", {}).get("k", DEFAULT_K)
    alpha = config.get("lfgd", {}).get("alpha", DEFAULT_ALPHA)
    tau = config.get("lfgd", {}).get("tau", DEFAULT_TAU)
    use_modal = config.get("retrieval", {}).get("use_modal", False)
    embedding_model = config.get("retrieval", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    use_embedding_cache = config.get("retrieval", {}).get("use_embedding_cache", True)
    embedding_cache_dir = Path(config.get("retrieval", {}).get("embedding_cache_dir", "data/cache"))
    reindex_qdrant = config.get("retrieval", {}).get("reindex_qdrant_on_eval", False)
    systems = config.get("systems", ["lfgd"])
    n_topics = config.get("n_topics")
    use_preprocessed_corpus = config.get("retrieval", {}).get("use_preprocessed_corpus", True)

    llm_config = config.get("llm", {})

    eval_topics = get_unique_topics(paired_docs, sort_by_frequency=True)
    if n_topics:
        eval_topics = eval_topics[:n_topics]

    all_texts: list[str] = []
    all_embeddings: np.ndarray | list = []
    all_sides: list[str] = []
    all_topics: list[str] = []

    corpus_loaded = False
    if use_preprocessed_corpus:
        corpus = _load_preprocessed_corpus()
        if corpus is not None:
            all_texts, all_embeddings, all_sides, all_topics = corpus
            corpus_loaded = True
            print(f"Loaded preprocessed retrieval corpus: {len(all_texts)} documents")

    if not corpus_loaded:
        # Fallback corpus from paired dataset rows.
        for paired in paired_docs:
            all_texts.append(paired.left)
            all_sides.append("l")
            all_topics.append(paired.topic)
            all_texts.append(paired.right)
            all_sides.append("r")
            all_topics.append(paired.topic)

    embedding_cache_path: Path | None = None
    query_cache_path: Path | None = None
    query_embeddings: np.ndarray | None = None
    model = None

    if use_embedding_cache:
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        model_key = _sanitize_model_name(embedding_model)
        doc_hash = _hash_text_list(all_texts)[:16]
        query_hash = _hash_text_list(eval_topics)[:16]
        embedding_cache_path = embedding_cache_dir / f"doc_emb_{model_key}_{doc_hash}.npy"
        query_cache_path = embedding_cache_dir / f"query_emb_{model_key}_{query_hash}.npy"

    # Embed all documents (skip if preprocessed embeddings were loaded)
    if not corpus_loaded:
        if embedding_cache_path is not None and embedding_cache_path.exists():
            all_embeddings = np.load(embedding_cache_path)
            print(f"Loaded cached document embeddings: {embedding_cache_path}")
        else:
            print(f"Embedding {len(all_texts)} documents...")
            if use_modal:
                import sys

                repo_root = Path(__file__).resolve().parents[1]
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))

                from modal_app import app, embed_texts

                with app.run():
                    all_embeddings = np.array(
                        embed_texts.remote(all_texts, embedding_model),
                        dtype=np.float32,
                    )
            else:
                from transformers.utils import logging as hf_logging
                from sentence_transformers import SentenceTransformer

                hf_logging.set_verbosity_error()
                model = SentenceTransformer(embedding_model)
                print("Computing local document embeddings (this can take several minutes)...")
                all_embeddings = model.encode(
                    all_texts,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )

            if embedding_cache_path is not None:
                np.save(embedding_cache_path, all_embeddings.astype(np.float32))
                print(f"Saved document embedding cache: {embedding_cache_path}")

    # Load or compute query embeddings
    if query_cache_path is not None and query_cache_path.exists():
        query_embeddings = np.load(query_cache_path)
        print(f"Loaded cached query embeddings: {query_cache_path}")
    else:
        questions = eval_topics
        if use_modal:
            import sys

            repo_root = Path(__file__).resolve().parents[1]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from modal_app import app, embed_texts

            with app.run():
                query_embeddings = np.array(
                    embed_texts.remote(questions, embedding_model),
                    dtype=np.float32,
                )
        else:
            if model is None:
                from transformers.utils import logging as hf_logging
                from sentence_transformers import SentenceTransformer

                hf_logging.set_verbosity_error()
                model = SentenceTransformer(embedding_model)
            print(f"Computing local query embeddings for {len(questions)} topics...")
            query_embeddings = model.encode(
                questions,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

        if query_cache_path is not None:
            np.save(query_cache_path, query_embeddings.astype(np.float32))
            print(f"Saved query embedding cache: {query_cache_path}")

    if query_embeddings is None:
        raise RuntimeError("Query embeddings were not prepared.")

    if all_embeddings.shape[0] != len(all_texts):
        raise ValueError(
            f"Document embedding count mismatch: expected {len(all_texts)}, got {all_embeddings.shape[0]}"
        )
    if query_embeddings.shape[0] != len(eval_topics):
        raise ValueError(
            f"Query embedding count mismatch: expected {len(eval_topics)}, got {query_embeddings.shape[0]}"
        )

    # Normalize embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_embeddings = all_embeddings / norms

    # Build retriever
    dense = build_retriever(
        all_texts,
        all_embeddings,
        use_hybrid=False,
        reindex_qdrant=reindex_qdrant,
    )
    if isinstance(dense, QdrantDenseRetriever):
        if reindex_qdrant:
            print(f"Using Qdrant dense retrieval (reindexed: {dense.collection_name})")
        else:
            print(f"Using Qdrant dense retrieval (reused: {dense.collection_name})")

    all_results: List[EvalResult] = []

    print(f"Running evaluation on {len(eval_topics)} topics...")
    for topic_idx, topic in enumerate(tqdm(eval_topics, desc="Processing topics")):
        question = topic  # In twinviews, topic is the question

        # Query embedding
        query_emb = query_embeddings[topic_idx]
        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0:
            continue
        query_emb = query_emb / q_norm

        # Retrieve top-N
        indices, scores = retrieve_candidates(dense, question, query_emb, N=N)

        # Build candidate docs with lean scores
        candidate_docs = []
        embeddings_list = [all_embeddings[i] for i in indices]
        embeddings_arr = np.stack(embeddings_list)

        # Compute lean scores using LFGD pipeline
        try:
            bias_axis = estimate_bias_axis_with_sign(embeddings_arr)
            lean_scores_arr, _ = compute_lean_scores_for_selection(
                embeddings_arr, bias_axis, tau=tau
            )
            if lean_scores_arr is None:
                lean_scores_arr = np.zeros(len(indices), dtype=float)
        except Exception:
            lean_scores_arr = np.zeros(len(indices))

        for i, idx in enumerate(indices):
            candidate_docs.append(
                RetrievedDoc(
                    text=all_texts[idx],
                    embedding=all_embeddings[idx],
                    relevance_score=float(scores[i]),
                    lean_score=float(lean_scores_arr[i]),
                    side=all_sides[idx],
                    topic=all_topics[idx],
                    doc_id=str(idx),
                )
            )

        # Evaluate
        query_results = evaluate_one_query(
            query_idx=len(all_results),
            topic=topic,
            question=question,
            candidate_docs=candidate_docs,
            systems=systems,
            llm_config=llm_config,
            k=K,
            alpha=alpha,
            tau=tau,
        )
        all_results.extend(query_results)

    return all_results


def write_results(results: List[EvalResult], output_dir: Path) -> None:
    """
    Write results to CSV and JSON files.

    Args:
        results: List of EvalResult objects
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results_{timestamp}"

    # Write CSV
    csv_path = output_dir / f"{base_name}.csv"
    fieldnames = [
        "topic", "system", "question", "selected_indices", "lean_scores",
        "cfair_score", "fair_score", "lean_var", "judge_category",
        "latency_sec", "synthesis", "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "topic": r.topic,
                "system": r.system,
                "question": r.question,
                "selected_indices": json.dumps(r.selected_indices),
                "lean_scores": json.dumps(r.lean_scores),
                "cfair_score": f"{r.cfair_score:.4f}",
                "fair_score": f"{r.fair_score:.4f}" if r.fair_score is not None else "",
                "lean_var": f"{r.lean_var:.4f}",
                "judge_category": r.judge_category,
                "latency_sec": f"{r.latency_sec:.2f}",
                "synthesis": r.synthesis,
                "error": r.error,
            })

    # Write JSON
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "topic": r.topic,
                    "system": r.system,
                    "question": r.question,
                    "selected_indices": r.selected_indices,
                    "lean_scores": r.lean_scores,
                    "cfair_score": r.cfair_score,
                    "fair_score": r.fair_score,
                    "lean_var": r.lean_var,
                    "judge_category": r.judge_category,
                    "latency_sec": r.latency_sec,
                    "synthesis": r.synthesis,
                    "error": r.error,
                }
                for r in results
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Results written to {output_dir}/")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LFGD evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=None,
        help="Number of topics to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.n_topics:
        config["n_topics"] = args.n_topics

    # Override output dir
    config["output_dir"] = Path(args.output_dir)

    # Load dataset
    print("Loading twinviews-13k dataset...")
    paired_docs = load_twinviews_huggingface()
    print(f"  Loaded {len(paired_docs)} paired documents")
    print(f"  {len(get_unique_topics(paired_docs))} unique topics")

    # Run experiment
    results = run_experiment(config, paired_docs)

    # Write results
    write_results(results, config["output_dir"])

    # Print summary
    print("\n=== Results Summary ===")
    systems = sorted(set(r.system for r in results))
    for system in systems:
        sys_results = [r for r in results if r.system == system]
        neutral_count = sum(1 for r in sys_results if r.judge_category == "Neutral")
        avg_cfair = np.mean([r.cfair_score for r in sys_results])
        fair_values = [r.fair_score for r in sys_results if r.fair_score is not None]
        avg_fair = np.mean(fair_values) if fair_values else float("nan")
        print(f"\n{system}:")
        print(f"  Neutral rate: {neutral_count}/{len(sys_results)} ({100*neutral_count/len(sys_results):.1f}%)")
        print(f"  Avg C-FAIR: {avg_cfair:.4f}")
        if not np.isnan(avg_fair):
            print(f"  Avg FAIR: {avg_fair:.4f}")


if __name__ == "__main__":
    main()

"""
Balanced set-selection for LFGD.

Implements exact combinatorial search over all C(N,k) subsets when
N <= 30 and k <= 8, with a greedy fallback for larger problems.
"""

from itertools import combinations
import numpy as np

from lfgd.objective import L_combined


def select_balanced_set_exact(
    lean_scores: np.ndarray,
    relevance_scores: np.ndarray,
    k: int,
    alpha: float = 0.5,
) -> list[int] | None:
    """
    Find the optimal k-subset using exact combinatorial search.

    Searches all C(N, k) subsets and selects the one minimizing
    L_combined(S) = α·L_utility(S) + (1-α)·L_fairness(S)

    Args:
        lean_scores: Array of shape (N,) with lean scores in [-1, 1]
        relevance_scores: Array of shape (N,) with relevance scores
        k: Number of documents to select (k <= N)
        alpha: Trade-off parameter (default 0.5)

    Returns:
        List of k indices representing the selected subset, or None if N > 30 or k > 8

    Note:
        Complexity: C(20, 6) = 38,760 iterations, <50ms on CPU
    """
    N = len(lean_scores)

    # Exact search only feasible for small N/k
    if N > 30 or k > 8:
        return None

    best_S = None
    best_loss = float("inf")

    for indices in combinations(range(N), k):
        indices_list = list(indices)
        loss = L_combined(indices_list, lean_scores, relevance_scores, alpha)
        if loss < best_loss:
            best_S = indices_list
            best_loss = loss

    return best_S


def select_balanced_set_greedy(
    lean_scores: np.ndarray,
    relevance_scores: np.ndarray,
    k: int,
    alpha: float = 0.5,
) -> list[int]:
    """
    Greedy selection as fallback for larger N or k.

    At each step, adds the document that produces the smallest
    marginal increase in the combined loss.

    Args:
        lean_scores: Array of shape (N,) with lean scores in [-1, 1]
        relevance_scores: Array of shape (N,) with relevance scores
        k: Number of documents to select
        alpha: Trade-off parameter (default 0.5)

    Returns:
        List of k indices representing the selected subset
    """
    N = len(lean_scores)
    selected: list[int] = []
    remaining = set(range(N))

    for _ in range(k):
        best_idx = None
        best_loss = float("inf")

        for idx in remaining:
            candidate = selected + [idx]
            loss = L_combined(candidate, lean_scores, relevance_scores, alpha)
            if loss < best_loss:
                best_idx = idx
                best_loss = loss

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def select_balanced_set(
    lean_scores: np.ndarray,
    relevance_scores: np.ndarray,
    k: int,
    alpha: float = 0.5,
) -> list[int]:
    """
    Select k documents balancing relevance and fairness.

    Tries exact search first (when N <= 30, k <= 8), falls back to
    greedy search for larger problems.

    Args:
        lean_scores: Array of shape (N,) with lean scores in [-1, 1]
        relevance_scores: Array of shape (N,) with relevance scores
        k: Number of documents to select
        alpha: Trade-off parameter (default 0.5)

    Returns:
        List of k indices representing the selected subset
    """
    N = len(lean_scores)

    if k > N:
        raise ValueError(f"Cannot select k={k} from N={N} candidates")

    if k == 0:
        return []

    # Try exact search first
    result = select_balanced_set_exact(lean_scores, relevance_scores, k, alpha)
    if result is not None:
        return result

    # Fall back to greedy
    return select_balanced_set_greedy(lean_scores, relevance_scores, k, alpha)


def select_top_k(
    relevance_scores: np.ndarray,
    k: int,
) -> list[int]:
    """
    Simple top-k selection based on relevance scores alone.

    Used as fallback when debiasing is not needed (variance gate fails).

    Args:
        relevance_scores: Array of shape (N,) with relevance scores
        k: Number of documents to select

    Returns:
        List of k indices sorted by relevance descending
    """
    indices = np.argsort(relevance_scores)[::-1][:k]
    return indices.tolist()
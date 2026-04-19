"""
Baseline rerankers for comparison with LFGD.

- unmitigated: Plain top-k by relevance (no fairness intervention)
- refarag: Probabilistic single-doc reranking (ReFaRAG)
- fairrag_select: BTP-I LLM set-selection (requires labels)
"""

from baselines.unmitigated import UnmitigatedResult, select_unmitigated
from baselines.refarag import ReFaRAGResult, refarag_rerank
from baselines.fairrag_select import FairRAGSelectResult, fairrag_select

__all__ = [
    "UnmitigatedResult",
    "select_unmitigated",
    "ReFaRAGResult",
    "refarag_rerank",
    "FairRAGSelectResult",
    "fairrag_select",
]

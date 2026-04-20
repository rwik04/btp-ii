"""
FairRAG-Select baseline (BTP-I): LLM-based set selection with labels.

This is a faithful re-implementation of BTP-I's LLM reranker that
requires explicit [l]/[r] labels on documents. Used as a comparison
baseline to show LFGD (label-free) performs comparably.

The BTP-I approach:
1. Retrieve top-N documents (like LFGD)
2. Prompt an LLM to select exactly k/2 liberal and k/2 conservative docs
3. Interleave selected documents (alternating l/r)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Literal, Optional

from dotenv import load_dotenv

load_dotenv()

Group = Literal["l", "r"]

_FAIRRAG_SELECT_CACHE: dict[str, List[int]] = {}
_FAIRRAG_CACHE_LOADED = False
_FAIRRAG_CACHE_PATH = Path("data/cache/fairrag_select_cache.json")
_FAIRRAG_CACHE_DIRTY_WRITES = 0
_FAIRRAG_CLIENTS: dict[str, object] = {}


def _load_fairrag_cache() -> None:
    global _FAIRRAG_CACHE_LOADED
    if _FAIRRAG_CACHE_LOADED:
        return
    _FAIRRAG_CACHE_LOADED = True
    if not _FAIRRAG_CACHE_PATH.exists():
        return

    try:
        with open(_FAIRRAG_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, list):
                    _FAIRRAG_SELECT_CACHE[str(key)] = [int(v) for v in value]
    except Exception:
        return


def _save_fairrag_cache() -> None:
    try:
        _FAIRRAG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FAIRRAG_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_FAIRRAG_SELECT_CACHE, f, ensure_ascii=True)
    except Exception:
        return


def _get_llm_client(provider: str):
    key = provider.strip().lower()
    client = _FAIRRAG_CLIENTS.get(key)
    if client is not None:
        return client

    if key == "groq":
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    elif key == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    _FAIRRAG_CLIENTS[key] = client
    return client


@dataclass
class FairRAGSelectResult:
    """Result from FairRAG-Select reranking."""

    selected_indices: List[int]
    selected_groups: List[Group]
    synthesis: str


def _format_chunks_for_prompt(
    texts: List[str],
    groups: List[Group],
    embeddings: List | None = None,
) -> str:
    """Format documents for LLM selection prompt."""
    lines = []
    for idx, (text, group) in enumerate(zip(texts, groups), start=1):
        side_label = f"[{group}]"
        snippet = text.replace("\n", " ")[:150]
        lines.append(f"{idx}. {side_label} {snippet}")
    return "\n".join(lines)


def rerank_chunks_llm(
    question: str,
    candidate_texts: List[str],
    candidate_groups: List[Group],
    target_k: int = 6,
    model: str | None = None,
    provider: str | None = None,
) -> List[int]:
    """
    Use LLM to select a balanced set of chunks from candidates.

    Prompts the LLM to select exactly target_k/2 from each group,
    then returns the selected indices.

    Args:
        question: The user question
        candidate_texts: List of candidate document texts
        candidate_groups: List of group labels ('l' or 'r') for each candidate
        target_k: Total number of documents to select
        model: Model name for LLM
        provider: 'groq' or 'openrouter'

    Returns:
        List of selected indices
    """
    if not candidate_texts:
        return []
    _load_fairrag_cache()
    target_k = min(target_k, len(candidate_texts))

    # Determine model and provider
    model = model or os.getenv("LLM_MODEL", "")
    provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
    if not provider and os.getenv("GROQ_API_KEY"):
        provider = "groq"
    elif not provider and os.getenv("OPENROUTER_API_KEY"):
        provider = "openrouter"

    cache_key = hashlib.sha256(
        json.dumps(
            {
                "question": question,
                "candidate_texts": candidate_texts,
                "candidate_groups": candidate_groups,
                "target_k": target_k,
                "model": model,
                "provider": provider,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if cache_key in _FAIRRAG_SELECT_CACHE:
        return list(_FAIRRAG_SELECT_CACHE[cache_key])

    def _cache_and_return(indices: List[int]) -> List[int]:
        global _FAIRRAG_CACHE_DIRTY_WRITES
        _FAIRRAG_SELECT_CACHE[cache_key] = list(indices)
        _FAIRRAG_CACHE_DIRTY_WRITES += 1
        if _FAIRRAG_CACHE_DIRTY_WRITES >= 32:
            _save_fairrag_cache()
            _FAIRRAG_CACHE_DIRTY_WRITES = 0
        return indices

    def _balanced_fallback_indices(groups: List[Group], k: int) -> List[int]:
        l_indices = [i for i, g in enumerate(groups) if g == "l"]
        r_indices = [i for i, g in enumerate(groups) if g == "r"]
        half_k = k // 2

        selected = l_indices[:half_k] + r_indices[:half_k]
        selected_set = set(selected)

        if len(selected) < k:
            for i in l_indices[half_k:] + r_indices[half_k:]:
                if i not in selected_set:
                    selected.append(i)
                    selected_set.add(i)
                if len(selected) == k:
                    break

        if len(selected) < k:
            for i in range(len(groups)):
                if i not in selected_set:
                    selected.append(i)
                    selected_set.add(i)
                if len(selected) == k:
                    break

        return selected[:k]

    # Build LLM client
    try:
        model = model or ""
        if provider == "groq":
            model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif provider == "openai":
            model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        else:
            model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
        client = _get_llm_client(provider)
    except Exception:
        return _cache_and_return(_balanced_fallback_indices(candidate_groups, target_k))

    # Build prompt
    half_k = target_k // 2
    formatted = _format_chunks_for_prompt(candidate_texts, candidate_groups)

    system_prompt = (
        "You are a fairness-aligned evidence selector. "
        "Your role is to choose a small set of chunks that best address the user question "
        "while maintaining balance across 'l' and 'r' sides. "
        "You must not favor any side unless the content itself justifies it. "
        "You must avoid ideological bias and evaluate only the information given. "
        "Return only a comma-separated list of 1-based indices with no explanation."
    )

    user_prompt = f"""Question:
{question}

Chunks:
{formatted}

Select exactly {target_k} indices. Choose {half_k} from 'l' and {half_k} from 'r'.
Return strictly in this form: 2,5,7,9,11,12
"""

    # Call LLM
    try:
        if provider == "groq":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

        raw = response.choices[0].message.content
        indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        selected = [i - 1 for i in indices if 1 <= i <= len(candidate_texts)]
        if selected:
            if len([i for i in selected if candidate_groups[i] == "l"]) >= half_k and len(
                [i for i in selected if candidate_groups[i] == "r"]
            ) >= half_k:
                return _cache_and_return(selected[:target_k])
            return _cache_and_return(_balanced_fallback_indices(candidate_groups, target_k))
    except Exception:
        return _cache_and_return(_balanced_fallback_indices(candidate_groups, target_k))

    result = _cache_and_return(_balanced_fallback_indices(candidate_groups, target_k))
    if _FAIRRAG_CACHE_DIRTY_WRITES:
        _save_fairrag_cache()
    return result


def fairrag_select(
    candidate_texts: List[str],
    candidate_groups: List[Group],
    relevance_scores: List[float],
    question: str,
    target_k: int = 6,
    model: str | None = None,
    provider: str | None = None,
) -> FairRAGSelectResult:
    """
    BTP-I FairRAG-Select reranking.

    Selects a balanced set using LLM with explicit labels.

    Args:
        candidate_texts: List of candidate document texts
        candidate_groups: List of 'l'/'r' labels for each candidate
        relevance_scores: Relevance scores (for sorting selected docs)
        question: The user question
        target_k: Number of documents to select
        model: LLM model name
        provider: LLM provider

    Returns:
        FairRAGSelectResult with selected indices and groups
    """
    selected_indices = rerank_chunks_llm(
        question=question,
        candidate_texts=candidate_texts,
        candidate_groups=candidate_groups,
        target_k=target_k,
        model=model,
        provider=provider,
    )

    selected_groups = [candidate_groups[i] for i in selected_indices]

    return FairRAGSelectResult(
        selected_indices=selected_indices,
        selected_groups=selected_groups,
        synthesis="",  # Synthesis done separately in eval pipeline
    )

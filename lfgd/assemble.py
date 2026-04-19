"""
Context assembly for LFGD.

Interleaves selected documents from both ends of the lean score spectrum,
ensuring the generator sees an alternating sequence of the two ideological poles.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ScoredDocument:
    """Document with associated lean and relevance scores."""
    text: str
    lean_score: float
    relevance_score: float
    metadata: dict | None = None


def interleave_by_lean(scored_docs: List[ScoredDocument]) -> List[ScoredDocument]:
    """
    Interleave documents from both ends of the lean score spectrum.

    Sorts documents by lean score ascending, then alternately takes from
    the high end (most positive) and low end (most negative), ensuring
    the generator sees an alternating sequence of opposing poles.

    Example for k=6:
        sorted:  [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
        output:  [0.6, -1.0, 0.2, -0.6, -0.2, 1.0]
                 or equivalently:
                 [most_pos, most_neg, second_pos, second_neg, ...]

    Args:
        scored_docs: List of ScoredDocument objects

    Returns:
        Interleaved list of ScoredDocument objects
    """
    if not scored_docs:
        return []

    # Sort by lean score ascending (most negative first)
    sorted_by_lean = sorted(scored_docs, key=lambda d: d.lean_score)

    interleaved: List[ScoredDocument] = []
    lo, hi = 0, len(sorted_by_lean) - 1
    mean_lean = sum(d.lean_score for d in sorted_by_lean) / len(sorted_by_lean)
    # If the set is slightly skewed to one side, start from the opposite pole
    # so earlier context positions are not consistently biased.
    start_positive = mean_lean <= 0

    while lo <= hi:
        if lo == hi:
            # Last remaining document
            interleaved.append(sorted_by_lean[lo])
            break

        if start_positive:
            # Take from high end (most positive lean) then low end.
            interleaved.append(sorted_by_lean[hi])
            interleaved.append(sorted_by_lean[lo])
        else:
            # Take from low end first when the set skews positive.
            interleaved.append(sorted_by_lean[lo])
            interleaved.append(sorted_by_lean[hi])

        lo += 1
        hi -= 1

    return interleaved


def assemble_context(
    selected_docs: List[ScoredDocument],
) -> str:
    """
    Assemble selected documents into a context string for generation.

    Applies lean-score interleaving then joins document texts.

    Args:
        selected_docs: List of ScoredDocument objects

    Returns:
        Context string with interleaved document texts
    """
    interleaved = interleave_by_lean(selected_docs)

    context_parts = []
    for i, doc in enumerate(interleaved, 1):
        # Include lean score indicator in context for transparency
        side = "[+]" if doc.lean_score > 0 else "[-]" if doc.lean_score < 0 else "[~]"
        context_parts.append(f"{i}. {side} {doc.text}")

    return "\n".join(context_parts)


def format_document_for_synthesis(
    doc: ScoredDocument,
    index: int,
) -> str:
    """
    Format a single document for synthesis prompt.

    Args:
        doc: ScoredDocument object
        index: 1-based index for display

    Returns:
        Formatted string like "1. [+] Some text..."
    """
    side = "[+]" if doc.lean_score > 0 else "[-]" if doc.lean_score < 0 else "[~]"
    text = doc.text.replace("\n", " ")
    return f"{index}. {side} {text}"

"""
Generation modules for LFGD synthesis and judging.

- generator: LLM synthesis from selected context
- judge: LLM 8-bin political lean classification
"""

from generation.generator import Generator, GenerationResult, format_context
from generation.judge import Judge, JudgeResult, JUDGE_CATEGORIES

__all__ = [
    "Generator",
    "GenerationResult",
    "format_context",
    "Judge",
    "JudgeResult",
    "JUDGE_CATEGORIES",
]

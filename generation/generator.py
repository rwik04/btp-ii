"""
LLM synthesis generator for LFGD responses.

Uses Groq or OpenRouter API to generate balanced synthesis
from the selected document context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# Judge categories (8-bin political lean)
JUDGE_CATEGORIES = [
    "Strongly Liberal",
    "Liberal",
    "Slightly Liberal",
    "Neutral",
    "Slightly Conservative",
    "Conservative",
    "Strongly Conservative",
    "Mixed/Unclear",
]


@dataclass
class GenerationResult:
    """Result of a synthesis generation."""

    synthesis: str
    model: str
    provider: str
    latency_sec: float


class Generator:
    """
    LLM-based synthesis generator.

    Supports Groq and OpenRouter providers.
    Generates neutral, balanced synthesis from document context.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ):
        """
        Initialize generator with LLM backend.

        Args:
            model: Model name (e.g. 'llama-3.1-8b-instant')
            provider: 'groq' or 'openrouter' (auto-detected if None)
            temperature: Sampling temperature (default 0.0 for deterministic)
            reasoning_effort: OpenAI reasoning effort ('low', 'medium', 'high')
                              for reasoning models (e.g. gpt-5*).
        """
        self.model = model or os.getenv("SYNTH_LLM_MODEL", "")
        self.provider = (provider or os.getenv("SYNTH_LLM_PROVIDER", "")).strip().lower()
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort or os.getenv("SYNTH_REASONING_EFFORT")
        self._llm = self._build_llm()

    def _build_llm(self):
        """Build the chat LLM backend."""
        # Try groq first if provider is groq or auto with GROQ_API_KEY
        if self.provider == "groq" or (
            not self.provider and os.getenv("GROQ_API_KEY")
        ):
            try:
                from groq import Groq

                model = self.model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                self._actual_model = model
                self._actual_provider = "groq"
                return client
            except ImportError:
                pass

        # Try OpenAI direct API (for gpt-5.4-mini etc.)
        if self.provider == "openai" or (
            not self.provider and os.getenv("OPENAI_API_KEY")
        ):
            try:
                from openai import OpenAI

                model = self.model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self._actual_model = model
                self._actual_provider = "openai"
                return client
            except ImportError:
                pass

        # Try OpenRouter (for gpt-oss-20b, gpt-oss-120b)
        try:
            from openai import OpenAI

            model = self.model or os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            self._actual_model = model
            self._actual_provider = "openrouter"
            return client
        except ImportError:
            pass

        raise RuntimeError("No LLM backend available. Set GROQ_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY.")

    def synthesize(
        self,
        question: str,
        context_docs: List[dict],
    ) -> GenerationResult:
        """
        Generate synthesis from selected context documents.

        Args:
            question: The user question
            context_docs: List of dicts with 'text', 'lean_score', 'relevance_score'

        Returns:
            GenerationResult with synthesis text and metadata
        """
        import time

        start = time.time()

        # Format context documents
        content_lines = []
        for i, doc in enumerate(context_docs, start=1):
            text = doc.get("text", "")
            content_lines.append(f"[{i}] {text}")

        content = "\n".join(content_lines)

        system_prompt = (
            "You are a neutral synthesis reporter. Your goal is to provide a crisp, objective "
            "summary of contrasting viewpoints based strictly on the provided context. "
            "Use neutral, descriptive framing and avoid any evaluative or loaded language. "
            "Acknowledge multiple perspectives symmetrically and attribute claims clearly."
        )

        user_prompt = f"""Question: {question}

Context:
{content}

Write a crisp 3-sentence synthesis (40-70 words total):
1. A single sentence stating the core tension or shared subject.
2. A single sentence contrasting the primary viewpoints with explicit attribution ('Some indicate... whereas others suggest...').
3. A single sentence summarizing the current state of evidence or common ground.

Constraints:
- No introductory filler (e.g., 'Based on the context...').
- No prescriptive language (should, must, need to).
- No qualitative labels (right, wrong, better, worse).
- Use dry, analytical reporting tone.
"""

        if self._actual_provider == "groq":
            response = self._llm.chat.completions.create(
                model=self._actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            synthesis = response.choices[0].message.content
        elif self._actual_provider == "openai":
            kwargs = {
                "model": self._actual_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            is_gpt5 = self._actual_model.lower().startswith("gpt-5")
            if not is_gpt5:
                kwargs["temperature"] = self.temperature
            if is_gpt5:
                kwargs["reasoning_effort"] = self.reasoning_effort or "medium"
            elif self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort
            response = self._llm.chat.completions.create(**kwargs)
            synthesis = response.choices[0].message.content
        else:
            response = self._llm.chat.completions.create(
                model=self._actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            synthesis = response.choices[0].message.content

        return GenerationResult(
            synthesis=synthesis,
            model=self._actual_model,
            provider=self._actual_provider,
            latency_sec=time.time() - start,
        )


def format_context_doc(doc: dict, index: int) -> str:
    """
    Format a single document for context.

    Args:
        doc: Document dict with 'text', 'side', 'lean_score', 'relevance_score'
        index: 1-based index

    Returns:
        Formatted string
    """
    text = doc.get("text", "")
    return f"[{index}] {text}"


def format_context(docs: List[dict]) -> str:
    """
    Format all documents for synthesis prompt.

    Args:
        docs: List of document dicts

    Returns:
        Formatted context string
    """
    lines = [format_context_doc(doc, i) for i, doc in enumerate(docs, start=1)]
    return "\n".join(lines)

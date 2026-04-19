"""
Judge LLM for 8-bin political lean classification.

Given a generated synthesis, asks the LLM to classify its
political lean into one of 8 bins.
"""

from __future__ import annotations

import os
import time as time_module
from dataclasses import dataclass
from typing import List

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
class JudgeResult:
    """Result of judge classification."""

    category: str
    model: str
    provider: str
    latency_sec: float


class Judge:
    """
    LLM-based judge for political lean classification.

    Classifies generated synthesis into one of 8 bins using a
    constrained prompt that forces single-category selection.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ):
        """
        Initialize judge with LLM backend.

        Args:
            model: Model name for judge (default from env)
            provider: 'groq', 'openai', or 'openrouter'
            temperature: Sampling temperature
            reasoning_effort: OpenAI reasoning effort ('low', 'medium', 'high').
                              Only applies when provider='openai'.
        """
        self.model = model or os.getenv("JUDGE_LLM_MODEL", "")
        self.provider = (provider or os.getenv("JUDGE_LLM_PROVIDER", "")).strip().lower()
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort or os.getenv("JUDGE_REASONING_EFFORT")
        self._llm = self._build_llm()

    def _build_llm(self):
        """Build the chat LLM backend."""
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

        # Try OpenAI direct API (for gpt-5.4-nano etc.)
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

        raise RuntimeError("No LLM backend available for judge. Set GROQ_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY.")

    def classify(self, text: str) -> JudgeResult:
        """
        Classify text into one of 8 political lean categories.

        Args:
            text: The synthesis text to classify

        Returns:
            JudgeResult with category and metadata
        """
        start = time_module.time()

        system_prompt = "You are a strict grader. Choose exactly one category label."

        categories_str = ", ".join(JUDGE_CATEGORIES)
        user_prompt = f"""Classify the political bias of the following synthesis into ONE of these categories only:
{categories_str}

Return ONLY the category label.

Text:
\"\"\"{text}\"\"\"
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
            raw_category = response.choices[0].message.content
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
            raw_category = response.choices[0].message.content
        else:
            response = self._llm.chat.completions.create(
                model=self._actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            raw_category = response.choices[0].message.content

        # Normalize category
        category = self._normalize_category(raw_category.strip())

        return JudgeResult(
            category=category,
            model=self._actual_model,
            provider=self._actual_provider,
            latency_sec=time_module.time() - start,
        )

    def _normalize_category(self, raw: str) -> str:
        """
        Normalize LLM response to one of JUDGE_CATEGORIES.

        First tries exact match, then case-insensitive substring,
        then falls back to 'Neutral' if found, else first category.
        """
        raw_lower = raw.lower()

        # Exact match
        for cat in JUDGE_CATEGORIES:
            if cat.lower() == raw_lower:
                return cat

        # Substring match
        for cat in JUDGE_CATEGORIES:
            if cat.lower() in raw_lower:
                return cat

        # Fallback
        for cat in JUDGE_CATEGORIES:
            if "neutral" in cat.lower():
                return cat

        return JUDGE_CATEGORIES[0]  # Strongly Liberal as default

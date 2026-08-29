"""Groq LLM client for Ask ControlPlane generative answer synthesis.

Thin wrapper around the Groq SDK.  Designed to be the single point of
contact with the external LLM API so that swapping providers later (e.g.
to OpenAI, Anthropic, or a local model) requires changing only this file.

Import-safe: if the ``groq`` package is not installed, the module still
imports without error -- ``GroqLLMClient.generate()`` raises a clear
``ImportError`` at call time, and ``is_available()`` returns ``False``,
letting the caller fall back to extractive synthesis gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag.config import rag_settings

logger = logging.getLogger("controlplane.rag.llm_client")

# ------------------------------------------------------------------
# Import guard — keep the module importable even without ``groq``
# ------------------------------------------------------------------
try:
    from groq import Groq  # type: ignore[import-untyped]
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    Groq = None  # type: ignore[assignment,misc]


_RAG_SYSTEM_PROMPT = (
    "You are ControlPlane AI, an internal assistant for enterprise AI "
    "governance, compliance, and company policy.  Answer the user's "
    "question using ONLY the provided context.  If the context does not "
    "contain enough information to answer confidently, say so — do NOT "
    "speculate or invent facts.  Be concise, accurate, and cite specific "
    "policy names or regulatory articles where relevant."
)


class GroqLLMClient:
    """Stateless, thread-safe wrapper for Groq chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else rag_settings.groq_api_key
        self.model = model if model is not None else rag_settings.groq_model
        self.max_tokens = max_tokens if max_tokens is not None else rag_settings.groq_max_tokens
        self.temperature = temperature if temperature is not None else rag_settings.groq_temperature

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """``True`` when the ``groq`` package is installed AND a key is configured."""
        return _GROQ_AVAILABLE and bool(rag_settings.groq_api_key)

    def generate(
        self,
        context: str,
        question: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a RAG-grounded prompt to Groq and return the generated answer.

        Raises
        ------
        ImportError
            If the ``groq`` package is not installed.
        RuntimeError
            If no API key is configured.
        groq.APIError (or subclass)
            On any API-level failure (rate limit, auth, server error).
        """
        if not _GROQ_AVAILABLE:
            raise ImportError(
                "The 'groq' package is not installed.  "
                "Install it with: pip install groq>=1.7.0"
            )
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.  Add it to your .env file or "
                "set the environment variable directly."
            )

        client = Groq(api_key=self.api_key)
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or _RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        return response.choices[0].message.content.strip()

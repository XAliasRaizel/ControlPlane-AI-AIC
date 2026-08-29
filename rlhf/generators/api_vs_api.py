"""ControlPlane.ai RLHF — API-vs-API pair generator.

Calls two API-hosted models concurrently and returns a ``PreferencePair``
that is ready to be labelled and stored.

Usage
-----
    import asyncio
    from rlhf.generators.api_vs_api import generate_api_vs_api_pair
    from rlhf.config import Category

    model_a = {"model_name": "gpt-4o-mini",  "temperature": 0.7, "top_p": 1.0}
    model_b = {"model_name": "claude-haiku", "temperature": 0.9, "top_p": 0.95}
    pair = asyncio.run(generate_api_vs_api_pair("Explain GDPR Art. 6", model_a, model_b))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from rlhf.config import Category, increment_generation_counter
from rlhf.schema import ModelResponse, PreferencePair
from rlhf.storage.categorize import assign_category

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub — wire this to the project's real API-calling function.
# ---------------------------------------------------------------------------

async def call_api_model(prompt: str, config: dict) -> str:
    """Call an API-hosted language model and return its text response.

    Args:
        prompt: The user prompt to send.
        config: Model configuration dict.  Expected keys:
            ``model_name`` (str), ``temperature`` (float, default 0.7),
            ``top_p`` (float, default 1.0), ``max_tokens`` (int, optional).

    Returns:
        The model's text response as a plain string.

    # TODO: wire this to the existing API-call function in the main codebase.
    #       In ControlPlane.ai that would be backend/shared/llm_simulator.py
    #       (for tests/local dev) or the real provider client used in
    #       backend/utils/llm_judge.py (OpenAIProvider / AnthropicProvider).
    #       Replace the NotImplementedError below with the actual call.
    """
    raise NotImplementedError(
        "call_api_model is a stub.  Wire it to your real LLM client before use."
    )


# ---------------------------------------------------------------------------
# Public async generator
# ---------------------------------------------------------------------------

async def generate_api_vs_api_pair(
    prompt: str,
    model_config_a: dict,
    model_config_b: dict,
    session_id: Optional[str] = None,
    category: Category = Category.UNSPECIFIED,
) -> PreferencePair:
    """Generate a preference pair by calling two API models concurrently.

    Both model calls are fired with ``asyncio.gather``.  If one call fails,
    the corresponding side of the pair is still returned with ``is_error=True``
    and ``error_message`` set — the pair is *never* silently dropped.

    Category is validated and attached via ``assign_category`` before the pair
    is returned, so callers can pass the pair straight to a storage backend.

    Args:
        prompt: The shared prompt sent to both models.
        model_config_a: Config dict for model A.  Must include
            ``"model_name"`` and may include ``"model_version_or_checkpoint"``,
            ``"temperature"``, ``"top_p"``, ``"max_tokens"``.
        model_config_b: Same shape as ``model_config_a`` for model B.
        session_id: Optional session correlation handle.
        category: Domain/category for the pair (``Category`` enum).

    Returns:
        A fully constructed ``PreferencePair`` (``chosen`` and ``labeled_by``
        will be None — this pair has not been labelled yet).

    Raises:
        RuntimeError: If the daily generation cap has been reached.
    """
    # Enforce daily cap BEFORE making any calls.
    increment_generation_counter()  # raises RuntimeError if cap reached
    increment_generation_counter()  # two calls in one pair

    async def _safe_call(config: dict) -> ModelResponse:
        """Wrap a single model call; capture errors gracefully."""
        model_name = config.get("model_name", "unknown")
        version = config.get("model_version_or_checkpoint", "unknown")
        hparams = {k: v for k, v in config.items()
                   if k not in ("model_name", "model_version_or_checkpoint")}
        try:
            text = await call_api_model(prompt, config)
            return ModelResponse(
                text=text,
                model_name=model_name,
                model_version_or_checkpoint=version,
                hyperparameters=hparams,
                is_error=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RLHF] api_vs_api call failed for model %s: %s", model_name, exc)
            return ModelResponse(
                text="",
                model_name=model_name,
                model_version_or_checkpoint=version,
                hyperparameters=hparams,
                is_error=True,
                error_message=str(exc),
            )

    response_a, response_b = await asyncio.gather(
        _safe_call(model_config_a),
        _safe_call(model_config_b),
    )

    pair = PreferencePair(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        session_id=session_id,
        source_pipeline="api_vs_api",
    )
    # Category is validated here — never allow raw strings past this point.
    pair = assign_category(pair, category)
    return pair

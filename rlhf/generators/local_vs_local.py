"""ControlPlane.ai RLHF — Local-vs-Local pair generator.

Mirrors ``generators/api_vs_api.py`` exactly, but targets locally-hosted
models (e.g. vLLM, Ollama, llama.cpp) rather than remote APIs.

Because both files return identical ``PreferencePair`` objects, you can
trivially produce *local-vs-API* pairs by calling one function from each
module and merging the two ``ModelResponse`` objects.  Example::

    from rlhf.generators.local_vs_local import call_local_model
    from rlhf.generators.api_vs_api    import call_api_model
    # …call both, build ModelResponse objects, construct PreferencePair manually.

Usage
-----
    import asyncio
    from rlhf.generators.local_vs_local import generate_local_vs_local_pair

    cfg_base = {"model_name": "llama-3-8b",      "temperature": 0.7}
    cfg_dpo  = {"model_name": "llama-3-8b-dpo",  "temperature": 0.7,
                "model_version_or_checkpoint": "./data/checkpoints/HR_20260830/"}
    pair = asyncio.run(generate_local_vs_local_pair("Explain leave policy", cfg_base, cfg_dpo))
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
# Stub — wire this to the project's real local-model-calling code.
# ---------------------------------------------------------------------------

async def call_local_model(prompt: str, config: dict) -> str:
    """Call a locally-hosted language model and return its text response.

    Args:
        prompt: The user prompt to send.
        config: Model configuration dict.  Expected keys:
            ``model_name`` (str), ``model_version_or_checkpoint`` (str,
            path to a local checkpoint or adapter), ``temperature`` (float),
            ``top_p`` (float), ``max_tokens`` (int).

    Returns:
        The model's text response as a plain string.

    # TODO: wire this to the existing local-model-calling code in the codebase.
    #       For ControlPlane.ai, that is ``backend/shared/gpu_adapter.py``
    #       (``GPUAdapter``) for real inference, or
    #       ``backend/shared/llm_simulator.py`` (``generate()``) for
    #       offline / test usage.  Replace the NotImplementedError below.
    #
    # NOTE: this same function shape can support local-vs-API pairs — just
    #       mix one call from this file and one from api_vs_api.py, then
    #       assemble a PreferencePair manually using both ModelResponse objects.
    """
    raise NotImplementedError(
        "call_local_model is a stub.  Wire it to your real local-model client before use."
    )


# ---------------------------------------------------------------------------
# Public async generator
# ---------------------------------------------------------------------------

async def generate_local_vs_local_pair(
    prompt: str,
    model_config_a: dict,
    model_config_b: dict,
    session_id: Optional[str] = None,
    category: Category = Category.UNSPECIFIED,
) -> PreferencePair:
    """Generate a preference pair by calling two local models concurrently.

    Identical guarantees to ``generate_api_vs_api_pair``:
    - Both calls are concurrent (``asyncio.gather``).
    - Each call is independently error-wrapped; a failed side sets
      ``is_error=True`` instead of dropping the pair.
    - Daily generation cap is enforced before any calls are made.
    - Category is validated and attached before the pair is returned.

    Args:
        prompt: The shared prompt sent to both models.
        model_config_a: Config dict for local model A.  Typically includes
            ``"model_name"``, ``"model_version_or_checkpoint"`` (checkpoint
            path), ``"temperature"``, ``"top_p"``.
        model_config_b: Same shape as ``model_config_a`` for model B.
        session_id: Optional session correlation handle.
        category: Domain/category for the pair (``Category`` enum).

    Returns:
        A fully constructed, unlabelled ``PreferencePair``.

    Raises:
        RuntimeError: If the daily generation cap has been reached.
    """
    # Enforce daily cap BEFORE making any calls.
    increment_generation_counter()
    increment_generation_counter()

    async def _safe_call(config: dict) -> ModelResponse:
        """Wrap a single local-model call; capture errors gracefully."""
        model_name = config.get("model_name", "unknown")
        version = config.get("model_version_or_checkpoint", "unknown")
        hparams = {k: v for k, v in config.items()
                   if k not in ("model_name", "model_version_or_checkpoint")}
        try:
            text = await call_local_model(prompt, config)
            return ModelResponse(
                text=text,
                model_name=model_name,
                model_version_or_checkpoint=version,
                hyperparameters=hparams,
                is_error=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[RLHF] local_vs_local call failed for model %s: %s", model_name, exc
            )
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
        source_pipeline="local_vs_local",
    )
    pair = assign_category(pair, category)
    return pair

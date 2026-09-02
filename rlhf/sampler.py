"""ControlPlane.ai RLHF -- Governance-pipeline preference-pair sampler.

This module is the integration point between the live governance pipeline
(backend/main.py) and the RLHF data-collection loop.

Design
------
* Called as a FastAPI BackgroundTask from /v1/govern -- never blocks
  the request path.
* 1 in every RLHF_SAMPLING_RATE_N requests (default: 10) triggers
  dual-response generation.
* Model A: Groq API (or simulator fallback) with temperature 0.7.
* Model B: llm_simulator.generate (deterministic, local). Gives us a
  real-vs-simulated pair useful for training even before a second real
  API model is added.
* The resulting pair is written to rlhf/data/raw/pairs.jsonl and
  labelled asynchronously by the configured LLM judge.
* ALL exceptions are silently swallowed -- a bug here must never
  affect governance request latency or correctness.

Category mapping
----------------
request.department maps to rlhf.config.Category:
  HR                              => Category.HR
  Finance / Financial / FINANCIAL => Category.FINANCIAL
  anything else                   => Category.GENERAL
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.shared.schemas import GovernanceRequest

logger = logging.getLogger("controlplane.rlhf.sampler")


# ---------------------------------------------------------------------------
# Department => Category mapping
# ---------------------------------------------------------------------------

def _infer_category(department):
    """Map a department string to an rlhf Category enum member.

    Args:
        department: The department field from a GovernanceRequest.

    Returns:
        A rlhf.config.Category enum member.
    """
    from rlhf.config import Category

    dept = (department or "").strip().upper()
    if dept == "HR":
        return Category.HR
    if dept in ("FINANCE", "FINANCIAL"):
        return Category.FINANCIAL
    return Category.GENERAL


# ---------------------------------------------------------------------------
# Model configs used for the two sides of each pair
# ---------------------------------------------------------------------------

_MODEL_A_CONFIG = {
    "model_name": "openai/gpt-oss-120b",
    "temperature": 0.7,
    "max_tokens": 512,
}

# Model B is the deterministic simulator -- always available, zero-cost.
_MODEL_B_CONFIG = {
    "model_name": "llm_simulator_v1",
    "temperature": 0.0,
    "max_tokens": 512,
}


# ---------------------------------------------------------------------------
# Async helpers (run inside an existing event loop as tasks)
# ---------------------------------------------------------------------------

async def _generate_and_store(prompt, session_id, category):
    """Generate a preference pair, write it, and queue it for human review.

    Args:
        prompt: The original governance request prompt.
        session_id: Optional session correlation handle.
        category: rlhf.config.Category enum member.
    """
    import asyncio
    import os
    from rlhf.generators.api_vs_api import generate_api_vs_api_pair
    from rlhf.storage import json_store

    try:
        pair = await generate_api_vs_api_pair(
            prompt=prompt,
            model_config_a=_MODEL_A_CONFIG,
            model_config_b=_MODEL_B_CONFIG,
            session_id=session_id,
            category=category,
        )
        json_store.write_pair(pair)
        logger.info(
            "[RLHF/sampler] pair %s written (category=%s, source=%s, awaiting human review)",
            pair.pair_id, category, pair.source_pipeline,
        )

        # By default, leave pair unlabelled so it directly populates the
        # "Human Labelling — Active Review" queue for human raters.
        # Set RLHF_AUTO_JUDGE_BACKGROUND=true only if automated machine labelling is desired.
        if os.getenv("RLHF_AUTO_JUDGE_BACKGROUND", "false").lower() == "true":
            await _judge_and_update(pair)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[RLHF/sampler] pair generation skipped: %s", exc)



async def _judge_and_update(pair):
    """Label a pair with the LLM judge and persist the label.

    Args:
        pair: An unlabelled PreferencePair from the JSONL store.
    """
    from rlhf.judges.llm_judge import judge_pair_with_llm
    from rlhf.storage import json_store

    try:
        labelled = judge_pair_with_llm(pair, n_calls=2)
        if labelled.chosen:
            json_store.update_label(
                pair_id=labelled.pair_id,
                chosen=labelled.chosen,
                labeled_by="llm_judge",
                judge_metadata=labelled.judge_metadata,
            )
            logger.info(
                "[RLHF/sampler] pair %s labelled '%s' by llm_judge",
                labelled.pair_id, labelled.chosen,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[RLHF/sampler] LLM judge labelling skipped: %s", exc)


# ---------------------------------------------------------------------------
# Public entry point -- called from backend/main.py as a BackgroundTask
# ---------------------------------------------------------------------------

async def maybe_collect_pair(request, candidate_response, context):
    """Possibly trigger RLHF preference-pair collection for this request.

    Called from /v1/govern, /v1/chat, and /v1/inspect as an async BackgroundTask.
    Generates dual-model responses (Groq API vs. Simulator) and queues them
    into the RLHF human review storage.

    All exceptions are silently swallowed -- a bug here must never
    affect the governance request.

    Args:
        request: The GovernanceRequest that was just processed.
        candidate_response: The response already generated (may be None
            if the request was BLOCKED).
        context: The enriched context dict (kept for future extensibility).
    """
    if not getattr(request, "prompt", None):
        return

    try:
        import os
        from rlhf.config import SAMPLING_RATE_N

        always_sample = os.getenv("RLHF_SAMPLE_ALL", "true").lower() == "true"
        if not always_sample and SAMPLING_RATE_N > 1:
            if random.random() >= 1.0 / SAMPLING_RATE_N:
                return  # Not sampled this turn.

        category = _infer_category(getattr(request, "department", None))
        session_id = getattr(request, "session_id", None)

        await _generate_and_store(request.prompt, session_id, category)

    except Exception as exc:  # noqa: BLE001
        logger.debug("[RLHF/sampler] maybe_collect_pair suppressed: %s", exc)


"""Async pipeline worker (Section 5.8).

Runs every async-only detector (hot_path=False) via the unified registry
and persists the result. v1 runs in-process; production would use Redis
Streams consumers pulling from the same DETECTOR_REGISTRY.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.shared.config import settings
from backend.shared.schemas import GovernanceRequest

logger = logging.getLogger("controlplane.async_pipeline")


import os
import random


async def process_async(
    request_id: str,
    request: GovernanceRequest,
    job_id: Optional[str] = None,
    hot_path_risk: Optional[float] = None,
) -> dict:
    """Run every async-path detector concurrently and persist the result.

    Smart sampling gate (configurable via env vars):
    - risk >= CONTROLPLANE_ASYNC_LLM_RISK_THRESHOLD (default 0.20) → run 100% of LLM checks
    - risk < threshold → sample CONTROLPLANE_ASYNC_LLM_SAMPLE_RATE (default 5%) for drift monitoring

    After analysis, corrective escalation: if any async engine returns score > 0.65
    (HIGH/CRITICAL risk) for a request the hot path ALLOWed, auto-enqueues to Human
    Review Queue and logs a warning for the SOC team.
    """
    from backend.async_pipeline.consumers import run_analytics_engines
    from backend.audit.store import Database

    effective_job_id = job_id or f"async-{request_id[:8]}"
    db = Database(settings.db_path)

    # Smart sampling gate (active when hot_path_risk is passed and sampling enabled)
    sampling_enabled = os.getenv("CONTROLPLANE_ASYNC_SAMPLING_ENABLED", "true").lower() == "true"
    risk_threshold = float(os.getenv("CONTROLPLANE_ASYNC_LLM_RISK_THRESHOLD", "0.20"))
    sample_rate = float(os.getenv("CONTROLPLANE_ASYNC_LLM_SAMPLE_RATE", "0.05"))

    if sampling_enabled and hot_path_risk is not None and hot_path_risk < risk_threshold:
        if random.random() > sample_rate:
            # Skip expensive LLM engines for this benign request
            logger.debug(
                "Async LLM analysis SAMPLED OUT for %s (risk=%.3f < %.2f threshold, rate=%.0f%%)",
                request_id, hot_path_risk, risk_threshold, sample_rate * 100,
            )
            try:
                db.update_job(effective_job_id, "SKIPPED_BENIGN", {
                    "request_id": request_id,
                    "job_id": effective_job_id,
                    "analytics": {},
                    "sampling_note": f"Benign request (risk={hot_path_risk:.3f}) sampled out at rate={sample_rate:.0%}",
                })
            except Exception:
                pass
            return {"request_id": request_id, "job_id": effective_job_id, "analytics": {}}

    try:
        analytics = await run_analytics_engines(request)
    except Exception as exc:
        logger.warning("Async analysis failed for %s: %s", request_id, exc)
        try:
            db.update_job(effective_job_id, "FAILED", {"error": str(exc)})
        except Exception as db_exc:
            logger.warning("Could not persist FAILED status for %s: %s", effective_job_id, db_exc)
        raise

    combined = {
        "request_id": request_id,
        "job_id": effective_job_id,
        "analytics": analytics,
    }

    try:
        job = db.get_job(effective_job_id)
        if job and job.get("result"):
            existing = job["result"]
            if isinstance(existing, dict):
                combined.update({k: v for k, v in existing.items() if k not in combined})
        db.update_job(effective_job_id, "COMPLETED", combined)
        logger.info("Async job %s persisted for request %s", effective_job_id, request_id)
    except Exception as exc:
        logger.warning("Could not update async job in DB: %s", exc)

    # Corrective escalation: if async LLM engines caught something the hot path missed,
    # enqueue to Human Review Queue and boost session risk for next turn.
    _run_corrective_escalation(request_id, request, analytics, hot_path_risk)

    logger.info("Async processing complete for %s", request_id)
    return combined


def _run_corrective_escalation(
    request_id: str,
    request: GovernanceRequest,
    analytics: dict,
    hot_path_risk: float,
) -> None:
    """Log to Human Review Queue when async LLM catches high-risk content the hot path missed."""
    HIGH_LABELS = {"HIGH", "CRITICAL", "BIASED", "INJECTION_DETECTED"}
    SCORE_THRESHOLD = 0.65

    flagged = [
        (name, result)
        for name, result in analytics.items()
        if (result.get("score", 0.0) > SCORE_THRESHOLD or result.get("status") in HIGH_LABELS)
        and result.get("status") not in {"SKIPPED_MOCK_PROVIDER", "DEGRADED", "MODEL_ERROR", "NOT_APPLICABLE"}
    ]

    if not flagged or hot_path_risk >= 0.70:
        # Either nothing flagged, or the hot path already caught it — no escalation needed
        return

    try:
        from backend.review.queue import ReviewQueue
        _queue = ReviewQueue()

        # Build a minimal synthetic decision to enqueue
        from backend.shared.schemas import GovernanceDecision
        corrective_decision = GovernanceDecision(
            action="HUMAN_REVIEW",
            policy_id="async-escalation",
            reason=(
                "Async LLM analysis detected high risk after hot-path ALLOW: "
                + "; ".join(f"{name}={res['status']}(score={res.get('score', 0):.2f})" for name, res in flagged)
            ),
            request_id=request_id,
        )
        _queue.enqueue(corrective_decision, prompt=request.prompt or "")
        logger.warning(
            "ASYNC_ESCALATION request_id=%s engines=%s — enqueued to Human Review",
            request_id,
            [name for name, _ in flagged],
        )
    except Exception as exc:
        logger.warning("Corrective escalation failed (non-blocking): %s", exc)

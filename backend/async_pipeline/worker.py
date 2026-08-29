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


async def process_async(
    request_id: str,
    request: GovernanceRequest,
    job_id: Optional[str] = None,
) -> dict:
    """Run every async-path detector concurrently and persist the result.

    FIX: this used to also loop over `DETECTOR_REGISTRY` for hot_path=False
    detectors in a sequential `for ... await` loop -- inconsistent with the
    hot path's own asyncio.gather pattern, and (because that registry slot
    was always empty before this fix) it never actually ran anything. The
    registry is now genuinely populated (see detectors/async_analytics.py),
    and run_analytics_engines() already fans out to all of it with gather,
    so that one call is the single source of async analysis -- no separate,
    duplicate loop needed here.

    FIX: this used to call run_analytics_engines() with no try/except. Any
    exception inside any one of the seven engines meant db.update_job() was
    never reached, leaving the job stuck at status="QUEUED" forever, with no
    way for the dashboard to tell "still working" apart from "silently
    failed." Failures are now caught and persisted as a genuine FAILED
    state instead.
    """
    from backend.async_pipeline.consumers import run_analytics_engines
    from backend.audit.store import Database

    effective_job_id = job_id or f"async-{request_id[:8]}"
    db = Database(settings.db_path)

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

    logger.info("Async processing complete for %s", request_id)
    return combined

"""Async pipeline worker (Section 5.8).

Runs async-only detectors (hot_path=False) and additional analytics
engines.  v1 uses in-process asyncio tasks; production would use Redis
Streams consumers.  Stores completed job results in the database.
"""

import asyncio
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
    """Run all async-path detectors + analytics engines and save result to database."""
    from backend.detectors.base import DETECTOR_REGISTRY
    from backend.async_pipeline.consumers import run_analytics_engines
    from backend.audit.store import Database

    # Run async-only detectors (hot_path=False)
    async_detectors = [d for d in DETECTOR_REGISTRY.values() if not d.hot_path]
    context: dict = {}
    detector_results = []
    for det in async_detectors:
        try:
            result = await det.analyze(request, context)
            detector_results.append(result.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Async detector %s failed: %s", det.name, exc)

    # Run additional analytics engines (safety, privacy, fairness, grounding, performance, cost, business)
    analytics = await run_analytics_engines(request)

    combined = {
        "request_id": request_id,
        "job_id": job_id or f"async-{request_id[:8]}",
        "async_detectors": detector_results,
        "analytics": analytics,
    }

    # Persist to database if job_id is provided or can be derived
    effective_job_id = job_id or f"async-{request_id[:8]}"
    try:
        db = Database(settings.db_path)
        db.update_job(effective_job_id, "COMPLETED", combined)
        logger.info("Async job %s persisted for request %s", effective_job_id, request_id)
    except Exception as exc:
        logger.warning("Could not update async job in DB: %s", exc)

    logger.info("Async processing complete for %s", request_id)
    return combined

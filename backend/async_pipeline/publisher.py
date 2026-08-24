"""Async pipeline publisher (Section 5.8).

Publishes governance events for async consumption.  v1 uses
asyncio.create_task; production would use Redis Streams.
"""

import asyncio
import logging
from typing import Optional

from backend.shared.schemas import GovernanceRequest

logger = logging.getLogger("controlplane.async_pipeline")


async def publish_event(
    request_id: str,
    request: GovernanceRequest,
    job_id: Optional[str] = None,
) -> None:
    """Fire-and-forget: schedule async analysis without blocking the response."""
    from backend.async_pipeline.worker import process_async

    effective_job_id = job_id or f"async-{request_id[:8]}"
    asyncio.create_task(process_async(request_id, request, effective_job_id))
    logger.debug("Async event published for request %s (job=%s)", request_id, effective_job_id)

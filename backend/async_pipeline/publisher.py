"""Async pipeline publisher (Section 5.8).

Publishes governance events for async consumption. v1 uses
asyncio.create_task; production would use Redis Streams.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from backend.shared.schemas import GovernanceRequest

logger = logging.getLogger("controlplane.async_pipeline")

# FIX: asyncio.create_task()'s return value was previously discarded.
# Per Python's own asyncio docs, the event loop only holds a *weak*
# reference to a task; a task with no other reference can be garbage
# collected mid-execution, silently, especially under real load in a
# long-running server (as opposed to a short-lived demo script, where
# this rarely has time to manifest). Keeping tasks in this module-level
# set -- and removing each one via a done-callback -- is the standard
# fix: https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_BACKGROUND_TASKS: set[asyncio.Task] = set()


async def publish_event(
    request_id: str,
    request: GovernanceRequest,
    job_id: Optional[str] = None,
) -> None:
    """Fire-and-forget: schedule async analysis without blocking the response."""
    from backend.async_pipeline.worker import process_async

    effective_job_id = job_id or f"async-{request_id[:8]}"
    task = asyncio.create_task(process_async(request_id, request, effective_job_id))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    logger.debug("Async event published for request %s (job=%s)", request_id, effective_job_id)

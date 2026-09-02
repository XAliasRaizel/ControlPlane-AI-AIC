"""Async pipeline publisher (Section 5.8).

Publishes governance events for async consumption. v1 uses
asyncio.create_task; production would use Redis Streams.

Reliability improvements:
- Dead-letter mechanism: failed async events are written to a
  `failed_async_jobs` SQLite table instead of being silently dropped.
  Use GET /v1/admin/dead-letters to inspect and retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Dead-letter store — SQLite table for failed async events
# ---------------------------------------------------------------------------
class DeadLetterStore:
    """Writes failed async events to `failed_async_jobs` so they are never
    silently dropped.  One connection per call (no pool needed here — only
    written on failures, which are rare)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_async_jobs (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    failed_at TEXT NOT NULL,
                    error TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    retried INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("DeadLetterStore: could not ensure table: %s", exc)

    def write(
        self,
        request_id: str,
        job_id: str,
        error: str,
        payload: dict,
    ) -> str:
        """Write a failed event. Returns the dead-letter record ID."""
        record_id = str(uuid.uuid4())
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute(
                """INSERT OR REPLACE INTO failed_async_jobs
                   (id, request_id, job_id, failed_at, error, payload, retried, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 0)""",
                (
                    record_id,
                    request_id,
                    job_id,
                    datetime.now(timezone.utc).isoformat(),
                    error,
                    json.dumps(payload, default=str),
                ),
            )
            conn.commit()
            conn.close()
            logger.warning(
                "Dead-letter written: request_id=%s job_id=%s id=%s",
                request_id, job_id, record_id,
            )
        except Exception as exc:
            logger.error(
                "CRITICAL: could not write to dead-letter store for request %s: %s",
                request_id, exc,
            )
        return record_id

    def list_all(self, limit: int = 100) -> list[dict]:
        """Return up to *limit* dead-letter records, newest first."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM failed_async_jobs ORDER BY failed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("DeadLetterStore.list_all failed: %s", exc)
            return []

    def mark_retried(self, record_id: str) -> None:
        """Mark a dead-letter record as retried."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute(
                "UPDATE failed_async_jobs SET retried=1, retry_count=retry_count+1 WHERE id=?",
                (record_id,),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("DeadLetterStore.mark_retried failed: %s", exc)

    def delete(self, record_id: str) -> None:
        """Remove a dead-letter record after successful manual retry."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("DELETE FROM failed_async_jobs WHERE id=?", (record_id,))
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("DeadLetterStore.delete failed: %s", exc)


# Module-level singleton — initialised on first publish_event call
_dead_letter_store: Optional[DeadLetterStore] = None


def _get_dead_letter_store() -> DeadLetterStore:
    global _dead_letter_store
    if _dead_letter_store is None:
        from backend.shared.config import settings
        _dead_letter_store = DeadLetterStore(settings.db_path)
    return _dead_letter_store


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------
async def publish_event(
    request_id: str,
    request: GovernanceRequest,
    job_id: Optional[str] = None,
    hot_path_risk: float = 0.0,
) -> None:
    """Fire-and-forget: schedule async analysis without blocking the response.

    On failure the event is written to the dead-letter store instead of
    being silently discarded.

    hot_path_risk: the overall_risk score from the synchronous hot path,
    forwarded to process_async for the smart sampling gate.
    """
    from backend.async_pipeline.worker import process_async

    effective_job_id = job_id or f"async-{request_id[:8]}"

    async def _run_with_dead_letter():
        try:
            await process_async(request_id, request, effective_job_id, hot_path_risk=hot_path_risk)
        except Exception as exc:
            logger.warning(
                "Async event FAILED for request %s (job=%s): %s — writing to dead-letter.",
                request_id, effective_job_id, exc,
            )
            _get_dead_letter_store().write(
                request_id=request_id,
                job_id=effective_job_id,
                error=f"{type(exc).__name__}: {exc}",
                payload={
                    "request_id": request_id,
                    "application_id": request.application_id,
                    "user_id": request.user_id,
                },
            )

    task = asyncio.create_task(_run_with_dead_letter())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    logger.debug(
        "Async event published for request %s (job=%s, hot_path_risk=%.3f)",
        request_id, effective_job_id, hot_path_risk,
    )

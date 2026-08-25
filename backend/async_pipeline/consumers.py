"""Async analytics engines (Section 5.8).

The seven engines themselves now live in backend/detectors/async_analytics.py
as registered BaseDetector plugins (hot_path=False), so DETECTOR_REGISTRY
actually contains the async detectors and process_async can run them with
asyncio.gather instead of a sequential loop.

run_analytics_engines() is kept, unchanged in its public shape, as a thin
adapter over those same instances -- the Streamlit dashboard and the
existing tests consume this exact {engine_name: {engine, score, evidence,
status}} dict shape, so nothing downstream needs to change.
"""

from __future__ import annotations

import asyncio

from backend.shared.schemas import GovernanceRequest


async def run_analytics_engines(request: GovernanceRequest) -> dict:
    """Run all async-only detectors concurrently; return the legacy dict shape."""
    from backend.detectors.base import DETECTOR_REGISTRY

    engines = [d for d in DETECTOR_REGISTRY.values() if not d.hot_path]
    results = await asyncio.gather(*[d.analyze(request, {}) for d in engines])
    return {
        r.detector_name: {
            "engine": r.detector_name,
            "score": r.score,
            "evidence": r.evidence,
            "status": r.label,
        }
        for r in results
    }

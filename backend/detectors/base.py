"""Detector plugin pattern (Section 7).

Every detector — hot-path or async — implements BaseDetector and
self-registers via the @register decorator.  Adding detector #10 is a
one-file change that never touches the gateway, risk engine, or decision
engine.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backend.shared.schemas import DetectorResult, GovernanceRequest

if TYPE_CHECKING:
    pass


class BaseDetector(ABC):
    """Interface every detector must implement."""

    name: str  # unique key, e.g. "pii"
    hot_path: bool = True  # False = async-only, for expensive detectors

    @abstractmethod
    async def analyze(
        self, request: GovernanceRequest, context: dict
    ) -> DetectorResult:
        ...


# ---------------------------------------------------------------------------
# Self-registration machinery
# ---------------------------------------------------------------------------
DETECTOR_REGISTRY: dict[str, BaseDetector] = {}


def register(detector_cls: type) -> type:
    """Class decorator that instantiates and registers a detector."""
    instance = detector_cls()
    DETECTOR_REGISTRY[instance.name] = instance
    return detector_cls


# ---------------------------------------------------------------------------
# Hot-path runner — actual parallelism via asyncio.gather
# ---------------------------------------------------------------------------
async def run_hot_path(
    request: GovernanceRequest, context: dict
) -> tuple[list[DetectorResult], float]:
    """Run all hot-path detectors concurrently and return results + elapsed ms."""
    detectors = [d for d in DETECTOR_REGISTRY.values() if d.hot_path]

    async def _timed(detector: BaseDetector) -> DetectorResult:
        start = time.perf_counter()
        result = await detector.analyze(request, context)
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = round(elapsed, 3)
        return result

    start = time.perf_counter()
    results = await asyncio.gather(*[_timed(d) for d in detectors])
    total_ms = (time.perf_counter() - start) * 1000
    return list(results), round(total_ms, 3)

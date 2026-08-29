import asyncio
import logging
from unittest.mock import patch

import pytest

from backend.main import run_fast_lane
from backend.shared.schemas import GovernanceRequest, DetectorResult
from backend.detectors import DETECTOR_REGISTRY


def test_fast_lane_timeout_fail_open(caplog):
    """Fast lane must fail open (not raise) when a detector exceeds the 250ms timeout."""

    class SlowDetector:
        fast_async = True
        name = "slow_detector"

        async def analyze(self, request, context):
            await asyncio.sleep(0.5)
            return DetectorResult(
                detector_name="slow", score=1.0, label="HIGH", confidence=1.0
            )

    with patch.dict(DETECTOR_REGISTRY, {"slow": SlowDetector()}):
        request = GovernanceRequest(
            request_id="test-req-123",
            user_id="test-user",
            application_id="test-app",
            prompt="Hello",
            response="Hello back",
        )

        with caplog.at_level(logging.INFO):
            # Use asyncio.run() — no pytest-asyncio needed
            asyncio.run(run_fast_lane(request, "job-123"))

    # The timeout must have been caught and logged
    assert "timeout=True" in caplog.text

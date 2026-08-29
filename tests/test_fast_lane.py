import asyncio
import pytest
import logging
from backend.main import run_fast_lane
from backend.shared.schemas import GovernanceRequest, DetectorResult
from backend.detectors import DETECTOR_REGISTRY
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_fast_lane_timeout_fail_open(caplog):
    # Setup a mock detector that sleeps longer than 250ms
    class SlowDetector:
        fast_async = True
        name = "slow_detector"
        async def analyze(self, request, context):
            await asyncio.sleep(0.5)
            return DetectorResult(detector_name="slow", score=1.0, label="HIGH", confidence=1.0)
            
    # Mock the registry
    with patch.dict(DETECTOR_REGISTRY, {"slow": SlowDetector()}):
        request = GovernanceRequest(
            request_id="test-req-123",
            prompt="Hello",
            response="Hello back"
        )
        
        with caplog.at_level(logging.INFO):
            # This should hit the 250ms timeout and fail open without raising an exception
            await run_fast_lane(request, "job-123")
        
        # Verify that the timeout was caught and logged
        assert "timeout=True" in caplog.text


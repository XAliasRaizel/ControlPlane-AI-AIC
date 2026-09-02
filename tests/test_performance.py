"""Performance / load tests for ControlPlane.ai.

These tests are SKIPPED by default and only run when:
    CONTROLPLANE_PERF_TESTS=1

This keeps the normal CI test pass fast. A separate CI job can opt-in
by setting the env var.

Run manually:
    CONTROLPLANE_PERF_TESTS=1 pytest tests/test_performance.py -v --tb=short

Requirements: httpx>=0.28.0 (already in requirements.txt)
"""
import asyncio
import os

# Prevent background RLHF sampler from firing live API calls during benchmarks
os.environ["RLHF_SAMPLE_ALL"] = "false"
os.environ["RLHF_SAMPLING_RATE_N"] = "999999"

import statistics
import time
from typing import List

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app

# ---------------------------------------------------------------------------
# Skip gate — opt-in only
# ---------------------------------------------------------------------------
PERF_ENABLED = os.environ.get("CONTROLPLANE_PERF_TESTS", "").strip() == "1"
skip_perf = pytest.mark.skipif(
    not PERF_ENABLED,
    reason="Performance tests skipped. Set CONTROLPLANE_PERF_TESTS=1 to enable.",
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
DEMO_HEADERS = {"X-API-Key": "demo-key-001"}
BASE_PAYLOAD = {
    "user_id": "perf-tester",
    "user_role": "employee",
    "department": "engineering",
    "application_id": "perf-bot",
    "prompt": "What is the standard process for requesting PTO?",
}

# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _async_govern(client: httpx.AsyncClient, payload: dict, headers: dict) -> float:
    """Fire one /v1/govern request and return wall-clock latency in seconds."""
    t0 = time.perf_counter()
    resp = await client.post("/v1/govern", json=payload, headers=headers, timeout=30.0)
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text[:200]}"
    return elapsed


async def _async_get(client: httpx.AsyncClient, path: str) -> float:
    """Fire one GET request and return wall-clock latency in seconds."""
    t0 = time.perf_counter()
    resp = await client.get(path, timeout=10.0)
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, f"Unexpected status {resp.status_code} on {path}"
    return elapsed


def p99(latencies: List[float]) -> float:
    """Return the 99th percentile latency from a list of seconds."""
    if len(latencies) < 2:
        return latencies[0] if latencies else 0.0
    return statistics.quantiles(sorted(latencies), n=100)[98]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_perf
def test_govern_p99_latency_under_10_concurrent():
    """10 simultaneous /v1/govern requests — p99 latency must be under 5 seconds.

    This covers cold-start overhead from ML model loading.
    The SLA here is generous (5s) because ML models may not be pre-warmed
    in the test environment. The important constraint is that the system
    does not hang or timeout under concurrent load.
    """
    with TestClient(app) as sync_client:
        # Warm up: one sequential request to load ML models
        sync_client.post("/v1/govern", json=BASE_PAYLOAD, headers=DEMO_HEADERS)

        async def run():
            # Use TestClient as ASGI transport for httpx
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                tasks = [
                    _async_govern(client, {**BASE_PAYLOAD, "user_id": f"perf-user-{i}"}, DEMO_HEADERS)
                    for i in range(10)
                ]
                return await asyncio.gather(*tasks)

        latencies = asyncio.run(run())
        p99_val = p99(list(latencies))
        print(f"  10-concurrent govern p99={p99_val*1000:.1f}ms  (max: 5000ms)")
        assert p99_val < 5.0, (
            f"p99 latency {p99_val*1000:.1f}ms exceeds 5000ms SLA. "
            f"All latencies: {[round(l*1000, 1) for l in latencies]}"
        )


@skip_perf
def test_govern_throughput_50_sequential():
    """50 sequential /v1/govern requests must complete in under 120 seconds.

    Sequential (not concurrent) so this primarily tests throughput and
    per-request overhead, not concurrency/queueing.
    Expected: ~1-3s per request (ML inference) = 50-150s. SLA: 120s total.
    """
    with TestClient(app) as sync_client:
        # Warm up
        sync_client.post("/v1/govern", json=BASE_PAYLOAD, headers=DEMO_HEADERS)

        total_start = time.perf_counter()
        latencies = []
        for i in range(50):
            t0 = time.perf_counter()
            resp = sync_client.post(
                "/v1/govern",
                json={**BASE_PAYLOAD, "user_id": f"throughput-user-{i}"},
                headers=DEMO_HEADERS,
            )
            latencies.append(time.perf_counter() - t0)
            assert resp.status_code == 200, f"Request {i} failed: {resp.status_code}"

        total_elapsed = time.perf_counter() - total_start
        avg_ms = statistics.mean(latencies) * 1000
        p99_ms = p99(latencies) * 1000
        print(f"  50-sequential govern: total={total_elapsed:.1f}s  avg={avg_ms:.1f}ms  p99={p99_ms:.1f}ms")
        assert total_elapsed < 120.0, (
            f"50 sequential requests took {total_elapsed:.1f}s — exceeds 120s threshold."
        )


@skip_perf
def test_metrics_endpoint_p99_under_100ms():
    """20 requests to GET /metrics — p99 must be under 100ms.

    The /metrics endpoint should be ultra-lightweight (just serialize
    in-memory Prometheus registry). No ML models, no DB queries.
    """
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Warm up endpoint once before measuring concurrent burst
            await client.get("/metrics")
            tasks = [_async_get(client, "/metrics") for _ in range(20)]
            return await asyncio.gather(*tasks)

    latencies = asyncio.run(run())
    p99_val = p99(list(latencies))
    print(f"  /metrics p99={p99_val*1000:.1f}ms  (max: 100ms)")
    assert p99_val < 0.100, (
        f"/metrics p99 {p99_val*1000:.1f}ms exceeds 100ms SLA. "
        f"All: {[round(l*1000, 1) for l in latencies]}"
    )


@skip_perf
def test_health_endpoint_under_50ms():
    """10 requests to GET /health — p99 must be under 50ms.

    /health is a basic liveness probe with no external calls.
    Sub-50ms is achievable even with Python overhead.
    """
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            tasks = [_async_get(client, "/health") for _ in range(10)]
            return await asyncio.gather(*tasks)

    latencies = asyncio.run(run())
    p99_val = p99(list(latencies))
    print(f"  /health p99={p99_val*1000:.1f}ms  (max: 50ms)")
    assert p99_val < 0.050, (
        f"/health p99 {p99_val*1000:.1f}ms exceeds 50ms SLA. "
        f"All: {[round(l*1000, 1) for l in latencies]}"
    )

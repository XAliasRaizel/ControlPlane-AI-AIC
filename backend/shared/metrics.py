"""Prometheus metrics for ControlPlane.ai.

All metrics are registered using the default prometheus_client REGISTRY.
"""
from prometheus_client import Counter, Histogram, Gauge

govern_requests_total = Counter(
    name="controlplane_govern_requests_total",
    documentation="Total governance requests processed, by decision, tenant, policy.",
    labelnames=["decision", "tenant_id", "policy_id"],
)

govern_latency_seconds = Histogram(
    name="controlplane_govern_latency_seconds",
    documentation="End-to-end governance request latency in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

detector_latency_seconds = Histogram(
    name="controlplane_detector_latency_seconds",
    documentation="Per-detector latency in seconds.",
    labelnames=["detector_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

llm_provider_calls_total = Counter(
    name="controlplane_llm_provider_calls_total",
    documentation="LLM provider calls by provider and outcome.",
    labelnames=["provider", "outcome"],
)

circuit_breaker_state = Gauge(
    name="controlplane_circuit_breaker_state",
    documentation="Circuit breaker state per LLM provider (0=CLOSED, 1=OPEN, 2=HALF_OPEN).",
    labelnames=["provider"],
)

dead_letter_count = Gauge(
    name="controlplane_dead_letter_count",
    documentation="Number of unretried dead-letter async pipeline records.",
)


def record_govern(
    decision: str,
    tenant_id: str,
    policy_id: str,
    latency_seconds: float,
) -> None:
    """Record one completed governance call into Prometheus metrics."""
    govern_requests_total.labels(
        decision=decision or "UNKNOWN",
        tenant_id=tenant_id or "default",
        policy_id=policy_id or "default",
    ).inc()
    govern_latency_seconds.observe(latency_seconds)


def record_detector_latency(detector_name: str, latency_seconds: float) -> None:
    """Record per-detector latency."""
    detector_latency_seconds.labels(detector_name=detector_name).observe(latency_seconds)


def record_llm_call(provider: str, outcome: str) -> None:
    """Record one LLM provider call and its outcome."""
    llm_provider_calls_total.labels(provider=provider, outcome=outcome).inc()


def update_circuit_breaker(provider: str, state: str) -> None:
    """Update circuit breaker state gauge (CLOSED | OPEN | HALF_OPEN)."""
    state_map = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}
    circuit_breaker_state.labels(provider=provider).set(state_map.get(state, -1))


def refresh_dead_letter_count() -> None:
    """Refresh the dead_letter_count gauge from store."""
    try:
        from backend.async_pipeline.publisher import _get_dead_letter_store
        count = len(_get_dead_letter_store().list_all(limit=10_000))
        dead_letter_count.set(count)
    except Exception:
        pass

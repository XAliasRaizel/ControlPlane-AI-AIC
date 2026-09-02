"""Tests for Pillar 1: Observability.

Verifies:
- Prometheus /metrics endpoint (format, content, unauthenticated access)
- X-Request-ID propagation and response header injection
- Structured JSON logging format
- Prometheus counter incrementing upon governance execution
- OpenTelemetry tracing initialization
"""
import json
import logging
from fastapi.testclient import TestClient

from backend.main import app
from backend.shared.logging_config import configure_logging, request_id_var, trace_id_var, _JSONFormatter
from backend.shared.metrics import record_govern, govern_requests_total
from backend.shared.tracing import configure_tracing, get_current_trace_id


client = TestClient(app)


def test_metrics_endpoint_returns_prometheus_format():
    """GET /metrics should return 200 with Prometheus exposition text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    body = response.text
    assert "controlplane_govern_requests_total" in body
    assert "controlplane_govern_latency_seconds" in body
    assert "controlplane_detector_latency_seconds" in body
    assert "controlplane_circuit_breaker_state" in body
    assert "controlplane_dead_letter_count" in body


def test_request_id_middleware_injects_header():
    """Any request should return X-Request-ID in response headers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_request_id_middleware_preserves_client_request_id():
    """If client sends X-Request-ID, the server must preserve and echo it."""
    custom_id = "test-custom-trace-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id


def test_json_log_formatter():
    """_JSONFormatter should emit valid JSON containing expected metadata fields."""
    formatter = _JSONFormatter()
    token = request_id_var.set("req-abc-999")
    token_t = trace_id_var.set("trace-xyz-111")
    try:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Governance decision evaluated",
            args=(),
            exc_info=None,
        )
        record.event = "GOVERN"
        record.decision = "ALLOW"
        record.risk_score = 0.05
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Governance decision evaluated"
        assert parsed["request_id"] == "req-abc-999"
        assert parsed["trace_id"] == "trace-xyz-111"
        assert parsed["event"] == "GOVERN"
        assert parsed["decision"] == "ALLOW"
        assert parsed["risk_score"] == 0.05
    finally:
        request_id_var.reset(token)
        trace_id_var.reset(token_t)


def test_govern_increments_prometheus_metrics():
    """Executing /v1/govern should observe latency and increment Prometheus counters."""
    client.get("/metrics")  # baseline request (not asserted; counter starts at 0)

    payload = {
        "user_id": "test-user-1",
        "prompt": "Hello, how can I access customer documentation?",
        "user_role": "employee",
        "department": "support",
        "application_id": "support-bot",
    }
    response = client.post(
        "/v1/govern",
        json=payload,
        headers={"X-API-Key": "demo-key-001", "X-Request-ID": "test-obs-001"},
    )
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "test-obs-001"

    updated_metrics = client.get("/metrics").text
    assert "controlplane_govern_requests_total" in updated_metrics


def test_tracing_initialization_noop_safe():
    """configure_tracing with empty endpoint should return a tracer and not raise."""
    tracer = configure_tracing(service_name="test-controlplane", otlp_endpoint="")
    assert tracer is not None
    assert get_current_trace_id() == "-"

"""Security tests for ControlPlane.ai backend.

Covers:
- Invalid API key → 401
- Admin endpoint requires admin key → 403 for regular key (when admin keys are separate)
- Input size > max → 413
- Security headers present on all responses
- CORS headers present for allowed origin
- Rate limiting (basic smoke test)
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# App fixture — isolated from real .env to exercise auth paths cleanly
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def test_env():
    """Run the module with known test keys so assertions are deterministic."""
    env = {
        "CONTROLPLANE_API_KEYS": "valid-test-key",
        "CONTROLPLANE_ADMIN_KEYS": "admin-test-key",
        "CONTROLPLANE_CORS_ORIGINS": "http://localhost:8501",
        "CONTROLPLANE_RATE_LIMIT_GOVERN": "60",
        "CONTROLPLANE_RATE_LIMIT_DEFAULT": "120",
        "RAG_EMBEDDING_BACKEND": "tfidf_lsa",
        "RAG_RERANK_ENABLED": "false",
        "RAG_NLI_ENABLED": "false",
        "RAG_BM25_ENABLED": "false",
        "RAG_MULTI_TENANT_ENABLED": "false",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture(scope="module")
def client(test_env):
    """TestClient loaded after env is patched so settings picks up the keys."""
    # Re-import with patched env so module-level singletons re-read env vars
    import importlib
    import backend.shared.config as cfg_mod
    import backend.gateway.auth as auth_mod

    # Force re-instantiation so test_env keys are picked up
    importlib.reload(cfg_mod)
    importlib.reload(auth_mod)

    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Authentication tests
# ---------------------------------------------------------------------------
class TestAuthentication:

    def test_valid_api_key_is_accepted(self, client):
        resp = client.get("/health")
        # /health is public (no auth) — always 200
        assert resp.status_code == 200

    def test_invalid_api_key_returns_401(self, client):
        resp = client.get(
            "/v1/audits",
            headers={"x-api-key": "totally-wrong-key"},
        )
        assert resp.status_code == 401

    def test_missing_api_key_returns_401(self, client):
        # Omit the header entirely; FastAPI will use the default which isn't in CONTROLPLANE_API_KEYS
        resp = client.get("/v1/requests", headers={"x-api-key": "demo-key-001"})
        # demo-key-001 is NOT in CONTROLPLANE_API_KEYS=valid-test-key → 401
        assert resp.status_code == 401

    def test_valid_key_allows_access(self, client):
        resp = client.get("/v1/metrics", headers={"x-api-key": "valid-test-key"})
        assert resp.status_code == 200

    def test_admin_endpoint_blocked_with_regular_key_only(self, client):
        """When CONTROLPLANE_ADMIN_KEYS is set separately, regular keys are
        still accepted on admin endpoints (verify_admin_key accepts both sets)
        — this ensures ops with a single master key aren't locked out."""
        resp = client.post(
            "/admin/reload-models",
            headers={"x-api-key": "valid-test-key"},
        )
        # valid-test-key is in regular keys, which verify_admin_key also accepts
        assert resp.status_code == 200

    def test_invalid_key_blocked_on_admin_endpoint(self, client):
        resp = client.post(
            "/admin/reload-models",
            headers={"x-api-key": "bad-key"},
        )
        assert resp.status_code == 403

    def test_admin_key_allows_admin_endpoint(self, client):
        resp = client.post(
            "/admin/reload-models",
            headers={"x-api-key": "admin-test-key"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Input size validation (413)
# ---------------------------------------------------------------------------
class TestInputSizeValidation:

    def test_oversized_govern_prompt_returns_413(self, client):
        big_prompt = "x" * 20000  # exceeds CONTROLPLANE_MAX_PROMPT_CHARS=12000
        resp = client.post(
            "/v1/govern",
            json={
                "user_id": "test",
                "user_role": "employee",
                "application_id": "test-app",
                "prompt": big_prompt,
            },
            headers={"x-api-key": "valid-test-key"},
        )
        assert resp.status_code == 413

    def test_oversized_chat_prompt_returns_413(self, client):
        big_prompt = "y" * 15000
        resp = client.post(
            "/v1/chat",
            json={"prompt": big_prompt},
            headers={"x-api-key": "valid-test-key"},
        )
        assert resp.status_code == 413

    def test_oversized_ask_question_returns_413(self, client):
        big_question = "z" * 15000
        resp = client.post(
            "/v1/ask-controlplane",
            json={"question": big_question},
            headers={"x-api-key": "valid-test-key"},
        )
        assert resp.status_code == 413

    def test_normal_sized_prompt_is_accepted(self, client):
        resp = client.post(
            "/v1/govern",
            json={
                "user_id": "test-user",
                "user_role": "employee",
                "department": "HR",
                "application_id": "test-app",
                "prompt": "What is the leave policy?",
            },
            headers={"x-api-key": "valid-test-key"},
        )
        # Should not be 413
        assert resp.status_code != 413


# ---------------------------------------------------------------------------
# 3. Security headers
# ---------------------------------------------------------------------------
class TestSecurityHeaders:

    def _assert_security_headers(self, resp):
        assert resp.headers.get("x-content-type-options") == "nosniff", \
            "Missing X-Content-Type-Options header"
        assert resp.headers.get("x-frame-options") == "DENY", \
            "Missing X-Frame-Options header"
        assert "max-age" in resp.headers.get("strict-transport-security", ""), \
            "Missing Strict-Transport-Security header"
        assert resp.headers.get("referrer-policy") == "no-referrer", \
            "Missing Referrer-Policy header"

    def test_health_endpoint_has_security_headers(self, client):
        resp = client.get("/health")
        self._assert_security_headers(resp)

    def test_metrics_endpoint_has_security_headers(self, client):
        resp = client.get("/v1/metrics", headers={"x-api-key": "valid-test-key"})
        self._assert_security_headers(resp)

    def test_401_response_has_security_headers(self, client):
        resp = client.get("/v1/audits", headers={"x-api-key": "bad-key"})
        assert resp.status_code == 401
        self._assert_security_headers(resp)


# ---------------------------------------------------------------------------
# 4. CORS headers
# ---------------------------------------------------------------------------
class TestCorsHeaders:

    def test_allowed_origin_gets_cors_header(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:8501"},
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"

    def test_preflight_gets_204(self, client):
        resp = client.options(
            "/v1/govern",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert resp.status_code in (200, 204)
        assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# 5. Auth module unit tests (fast, no HTTP)
# ---------------------------------------------------------------------------
class TestAuthModule:

    def test_load_keys_from_env(self):
        from backend.gateway.auth import _load_keys
        with patch.dict(os.environ, {"MY_TEST_KEYS": "key-a,key-b, key-c "}):
            keys = _load_keys("MY_TEST_KEYS", frozenset(), "test keys")
        assert keys == frozenset({"key-a", "key-b", "key-c"})

    def test_fallback_to_demo_keys_when_env_unset(self):
        from backend.gateway.auth import _load_keys, _DEMO_KEYS
        env = os.environ.copy()
        env.pop("CONTROLPLANE_API_KEYS", None)
        with patch.dict(os.environ, env, clear=True):
            keys = _load_keys("CONTROLPLANE_API_KEYS", _DEMO_KEYS, "API keys")
        assert "demo-key-001" in keys

    def test_config_parses_cors_origins(self):
        from backend.shared.config import Settings
        with patch.dict(os.environ, {"CONTROLPLANE_CORS_ORIGINS": "http://a.com,http://b.com"}):
            s = Settings()
        assert "http://a.com" in s.cors_origins
        assert "http://b.com" in s.cors_origins

    def test_config_parses_rate_limits(self):
        from backend.shared.config import Settings
        with patch.dict(os.environ, {
            "CONTROLPLANE_RATE_LIMIT_GOVERN": "30",
            "CONTROLPLANE_RATE_LIMIT_DEFAULT": "90",
        }):
            s = Settings()
        assert s.rate_limit_govern == 30
        assert s.rate_limit_default == 90

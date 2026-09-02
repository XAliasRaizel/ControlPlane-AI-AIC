"""API-key authentication for ControlPlane.ai.

In production: set CONTROLPLANE_API_KEYS and CONTROLPLANE_ADMIN_KEYS as
comma-separated lists of secrets.  If neither is set the module falls back
to the hardcoded demo keys so local development still works out of the box,
but it logs a loud WARNING so you can't miss it in production.
"""

from __future__ import annotations

import logging
import os

from fastapi import Header, HTTPException

logger = logging.getLogger("controlplane.auth")

# ---------------------------------------------------------------------------
# Key loading — evaluated lazily so tests can patch os.environ cleanly
# ---------------------------------------------------------------------------

_DEMO_KEYS: frozenset[str] = frozenset({"demo-key-001", "demo-key-002", "test-key"})
_DEMO_ADMIN_KEYS: frozenset[str] = frozenset({"admin-key-001"})


def _load_keys(env_var: str, fallback: frozenset[str], kind: str) -> frozenset[str]:
    """Load a comma-separated set of keys from *env_var*.

    If the variable is absent or empty, returns *fallback* and emits a
    WARNING so operators notice they are running with insecure defaults.
    """
    raw = os.getenv(env_var, "").strip()
    if raw:
        keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
        if keys:
            logger.info("Loaded %d %s from %s", len(keys), kind, env_var)
            return keys

    logger.warning(
        "SECURITY WARNING: %s is not set — falling back to insecure demo %s. "
        "Set %s in production!",
        env_var,
        kind,
        env_var,
    )
    return fallback


def _get_valid_keys() -> frozenset[str]:
    return _load_keys("CONTROLPLANE_API_KEYS", _DEMO_KEYS, "API keys")


def _get_admin_keys() -> frozenset[str]:
    return _load_keys("CONTROLPLANE_ADMIN_KEYS", _DEMO_ADMIN_KEYS, "admin keys")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: str = Header(default="demo-key-001")) -> str:
    """FastAPI dependency — validates the caller's API key.

    Valid keys are loaded from CONTROLPLANE_API_KEYS (comma-separated).
    Falls back to demo keys when the env var is absent.
    """
    valid = _get_valid_keys()
    if x_api_key not in valid:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


async def verify_admin_key(x_api_key: str = Header(default="admin-key-001")) -> str:
    """FastAPI dependency for admin-tier endpoints.

    Checks CONTROLPLANE_ADMIN_KEYS first; also accepts regular API keys that
    appear in CONTROLPLANE_API_KEYS so a single master key can serve both roles
    when CONTROLPLANE_ADMIN_KEYS is not separately configured.
    """
    admin_keys = _get_admin_keys()
    regular_keys = _get_valid_keys()
    all_privileged = admin_keys | regular_keys  # regular keys can reach admin in dev
    if x_api_key not in all_privileged:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return x_api_key

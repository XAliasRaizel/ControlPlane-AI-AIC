"""Session state storage for the Session Risk Accumulator.

Every method must fail closed to "no session state" rather than raising into
the hot/fast path -- mirrors model_backend.py's consult() fail-safe philosophy.

CONCURRENCY CAVEAT: InMemorySessionStore does NOT share state across worker
processes. If the deployment runs more than one worker, sessions whose turns
land on different workers will under-count. This is a known, documented
limitation of the default store -- not a silent gap. If
CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED=true and CONTROLPLANE_SESSION_STORE
is unset, a warning is logged at startup recommending Redis for multi-worker
deployments.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

log = logging.getLogger("controlplane.session_store")


# ---------------------------------------------------------------------------
# Session state dataclass
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Mutable state accumulated across turns within one session."""

    session_id: str
    created_at: float
    last_updated_at: float
    ewma_score: float = 0.0
    peak_score: float = 0.0
    turn_count: int = 0
    fragment_window: list = field(default_factory=list)   # rolling PII fragments
    contaminated_tools: list = field(default_factory=list)
    contamination_active: bool = False
    fast_lane_correction_count: int = 0
    last_band: int = 1  # 1=baseline, 2=elevated, 3=high

    @property
    def session_risk(self) -> float:
        """The fused dual-signal risk score: max of EWMA and peak-with-decay."""
        return max(self.ewma_score, self.peak_score)

    def to_json(self) -> str:
        """Serialise to JSON string for Redis storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SessionState":
        """Deserialise from JSON string produced by to_json()."""
        data = json.loads(raw)
        return cls(**data)


# ---------------------------------------------------------------------------
# Store protocol + implementations
# ---------------------------------------------------------------------------

class SessionStore(Protocol):
    """Structural protocol — any class with these three methods qualifies."""

    def get(self, session_id: str) -> SessionState | None: ...
    def set(self, session_id: str, state: SessionState, ttl_seconds: int) -> None: ...
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Zero-dependency default.

    Not shared across processes — see the module docstring concurrency caveat.
    This is the correct default for single-process demos and tests; for
    multi-worker production use CONTROLPLANE_SESSION_STORE=redis://...
    """

    def __init__(self) -> None:
        # id -> (state, expires_at)
        self._data: dict[str, tuple[SessionState, float]] = {}

    def get(self, session_id: str) -> SessionState | None:
        entry = self._data.get(session_id)
        if entry is None:
            return None
        state, expires_at = entry
        if time.time() > expires_at:
            del self._data[session_id]
            return None
        return state

    def set(self, session_id: str, state: SessionState, ttl_seconds: int) -> None:
        self._data[session_id] = (state, time.time() + ttl_seconds)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionStore:
    """Optional Redis-backed store.

    Only imports redis lazily inside __init__ -- never at module load -- same
    lazy-import discipline used for torch/transformers elsewhere in this repo.
    If the redis import fails or connection fails, __init__ raises so that
    get_session_store() can log and fall back to InMemorySessionStore rather
    than crashing the request path.
    """

    def __init__(self, url: str) -> None:
        import redis  # lazy import -- NOT at module level
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()  # fail fast at construction, not at first request

    def _key(self, session_id: str) -> str:
        return f"cp:session:{session_id}"

    def get(self, session_id: str) -> SessionState | None:
        try:
            raw = self._client.get(self._key(session_id))
            return SessionState.from_json(raw) if raw else None
        except Exception:
            return None

    def set(self, session_id: str, state: SessionState, ttl_seconds: int) -> None:
        try:
            self._client.set(self._key(session_id), state.to_json(), ex=ttl_seconds)
        except Exception:
            pass  # fail closed -- next turn will recreate state

    def delete(self, session_id: str) -> None:
        try:
            self._client.delete(self._key(session_id))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton factory — mirrors get_detector_model() in model_backend.py
# ---------------------------------------------------------------------------

_store_singleton: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Seam entry point. Caches a singleton.

    Reads CONTROLPLANE_SESSION_STORE env var (a Redis URL). On any Redis
    construction failure, logs a warning and falls back to InMemorySessionStore
    -- never raises out of this function.
    """
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    url = os.environ.get("CONTROLPLANE_SESSION_STORE", "").strip()
    if url:
        try:
            _store_singleton = RedisSessionStore(url)
            log.info("SessionStore: using Redis at %s", url)
            return _store_singleton
        except Exception as exc:
            log.warning(
                "Redis session store unavailable (%s), falling back to in-memory. "
                "Sessions will not be shared across worker processes.",
                exc,
            )

    _store_singleton = InMemorySessionStore()
    log.info(
        "SessionStore: using InMemorySessionStore. "
        "NOTE: state is NOT shared across worker processes. "
        "Set CONTROLPLANE_SESSION_STORE=redis://... for multi-worker deployments."
    )
    return _store_singleton


def reset_store_cache() -> None:
    """For tests -- clears the singleton. Mirrors model_backend.py reset_cache()."""
    global _store_singleton
    _store_singleton = None

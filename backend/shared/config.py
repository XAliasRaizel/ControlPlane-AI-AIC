"""Centralised application settings.

All configuration is resolved from environment variables with sensible
defaults for local development.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]  # repo root

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    _env_file = _ROOT / ".env"
    if _env_file.exists():
        with open(_env_file, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k.strip() not in os.environ:
                        os.environ[_k.strip()] = _v.strip()

DEFAULT_POLICIES_DIR = _ROOT / "policies"
DEFAULT_DB_PATH = "controlplane.db"


def _parse_list(env_var: str, default: str = "") -> list[str]:
    """Parse a comma-separated env var into a stripped list."""
    raw = os.getenv(env_var, default).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    db_path: str = field(
        default_factory=lambda: os.getenv("CONTROLPLANE_DB_PATH", DEFAULT_DB_PATH)
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("CONTROLPLANE_LOG_LEVEL", "INFO")
    )
    async_delay_ms: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_ASYNC_DELAY_MS", "50"))
    )
    policies_dir: str = field(
        default_factory=lambda: os.getenv("CONTROLPLANE_POLICIES_DIR", str(DEFAULT_POLICIES_DIR))
    )
    audit_hash_key: str = field(
        default_factory=lambda: os.getenv(
            "CONTROLPLANE_AUDIT_HASH_KEY", "local-prototype-not-a-secret"
        )
    )
    max_prompt_chars: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_MAX_PROMPT_CHARS", "12000"))
    )

    # --- Security: API key sets ---
    # Comma-separated lists. If unset, auth.py falls back to demo keys with a WARNING.
    api_keys: list = field(
        default_factory=lambda: _parse_list("CONTROLPLANE_API_KEYS")
    )
    admin_keys: list = field(
        default_factory=lambda: _parse_list("CONTROLPLANE_ADMIN_KEYS")
    )

    # --- Security: CORS ---
    # Comma-separated allowed origins for CORS. Defaults to local Streamlit frontend.
    cors_origins: list = field(
        default_factory=lambda: _parse_list(
            "CONTROLPLANE_CORS_ORIGINS", "http://localhost:8501"
        )
    )

    # --- Security: Rate limiting ---
    # Requests per minute on /v1/govern (heavy governance endpoint).
    rate_limit_govern: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_RATE_LIMIT_GOVERN", "60"))
    )
    # Requests per minute on all other endpoints.
    rate_limit_default: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_RATE_LIMIT_DEFAULT", "120"))
    )

    # --- Scalability: Session TTL ---
    # How many hours a session is kept before SQLiteSessionStore vacuum removes it.
    session_ttl_hours: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_SESSION_TTL_HOURS", "24"))
    )

    # --- Scalability: Async task queue ---
    # Max items buffered in the asyncio.Queue before back-pressure is applied.
    async_queue_size: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_ASYNC_QUEUE_SIZE", "500"))
    )

    # --- Scalability: Metrics TTL cache ---
    # Seconds to cache /v1/metrics and /v1/audits results (avoids repeated heavy scans).
    metrics_cache_ttl_s: int = field(
        default_factory=lambda: int(os.getenv("CONTROLPLANE_METRICS_CACHE_TTL_S", "60"))
    )

    # Backward compat: single policy_path still works (points into policies_dir)
    @property
    def policy_path(self) -> str:
        return os.getenv(
            "CONTROLPLANE_POLICY_PATH",
            str(Path(self.policies_dir) / "global.yaml"),
        )


settings = Settings()

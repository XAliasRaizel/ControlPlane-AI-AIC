"""Centralised application settings.

All configuration is resolved from environment variables with sensible
defaults for local development.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]  # repo root

DEFAULT_POLICIES_DIR = _ROOT / "policies"
DEFAULT_DB_PATH = "controlplane.db"


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("CONTROLPLANE_DB_PATH", DEFAULT_DB_PATH)
    log_level: str = os.getenv("CONTROLPLANE_LOG_LEVEL", "INFO")
    async_delay_ms: int = int(os.getenv("CONTROLPLANE_ASYNC_DELAY_MS", "50"))
    policies_dir: str = os.getenv("CONTROLPLANE_POLICIES_DIR", str(DEFAULT_POLICIES_DIR))
    audit_hash_key: str = os.getenv(
        "CONTROLPLANE_AUDIT_HASH_KEY", "local-prototype-not-a-secret"
    )
    max_prompt_chars: int = int(os.getenv("CONTROLPLANE_MAX_PROMPT_CHARS", "12000"))

    # Backward compat: single policy_path still works (points into policies_dir)
    @property
    def policy_path(self) -> str:
        return os.getenv(
            "CONTROLPLANE_POLICY_PATH",
            str(Path(self.policies_dir) / "global.yaml"),
        )


settings = Settings()

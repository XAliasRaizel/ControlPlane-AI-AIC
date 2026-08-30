"""ControlPlane.ai RLHF — Central Configuration.

All constants consumed across the rlhf/ package live here.  Import
nothing from the rest of the codebase; this file must be importable
in complete isolation.
"""

from __future__ import annotations

import datetime
import json
import os
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Repo-relative root so paths resolve correctly regardless of CWD.
# ---------------------------------------------------------------------------
_RLHF_ROOT = Path(__file__).resolve().parent          # …/rlhf/
_PROJECT_ROOT = _RLHF_ROOT.parent                     # repo root


# ---------------------------------------------------------------------------
# 2.  Category enum — closed, validated set of domains a pair can belong to.
#
#     Extend here when a new model / domain is onboarded.  Never pass a raw
#     string through the system — always use this enum.  storage/categorize.py
#     is the enforcement point at write time.
# ---------------------------------------------------------------------------
class Category(str, Enum):
    """Downstream model / domain a preference pair belongs to.

    Using ``str`` as a mixin makes the enum JSON-serialisable without a
    custom encoder — ``Category.HR`` serialises as the string ``"HR"``.
    """

    HR = "HR"
    FINANCIAL = "FINANCIAL"
    GENERAL = "GENERAL"
    UNSPECIFIED = "UNSPECIFIED"
    # Add future domains below, e.g.:
    # LEGAL = "LEGAL"
    # MEDICAL = "MEDICAL"


# ---------------------------------------------------------------------------
# 3.  Storage backend selector.
#
#     Change this one string to "sqlite" to activate the SQLite backend
#     (once it has been verified against the test suite).  Every file that
#     needs a store reads THIS constant rather than hard-coding a backend.
# ---------------------------------------------------------------------------
STORAGE_BACKEND: str = os.getenv("RLHF_STORAGE_BACKEND", "json")  # "json" | "sqlite"


# ---------------------------------------------------------------------------
# 4.  Sampling / rate-limiting constants.
# ---------------------------------------------------------------------------
SAMPLING_RATE_N: int = int(os.getenv("RLHF_SAMPLING_RATE_N", "1"))
"""One in every N prompts triggers dual-generation (or 1/N probability)."""


MAX_DAILY_JUDGE_CALLS: int = int(os.getenv("RLHF_MAX_DAILY_JUDGE_CALLS", "200"))
"""Hard cap on LLM judge calls per calendar day (cost-control safety net)."""

MAX_DAILY_GENERATION_CALLS: int = int(os.getenv("RLHF_MAX_DAILY_GENERATION_CALLS", "500"))
"""Hard cap on dual-generation calls per calendar day."""


# ---------------------------------------------------------------------------
# 5.  File paths.
# ---------------------------------------------------------------------------
_DATA_DIR = _RLHF_ROOT / "data"

RAW_JSONL_PATH: Path = _DATA_DIR / "raw" / "pairs.jsonl"
SQLITE_DB_PATH: Path = _DATA_DIR / "db" / "preferences.db"
EXPORTS_DIR: Path = _DATA_DIR / "exports"
CHECKPOINTS_DIR: Path = _DATA_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# 6.  In-memory daily-call counters with file-based persistence.
#
#     Design intent:
#       • Simple daily reset — counter file is stamped with today's date;
#         if the date changes, the counter is discarded.
#       • NOT a production rate-limiter (no locking, no Redis).  This is a
#         cheap hackathon-appropriate safety net to avoid accidental runaway
#         API cost.
# ---------------------------------------------------------------------------
_COUNTER_DIR = _DATA_DIR / "counters"
_COUNTER_DIR.mkdir(parents=True, exist_ok=True)

_JUDGE_COUNTER_FILE = _COUNTER_DIR / "judge_calls.json"
_GEN_COUNTER_FILE = _COUNTER_DIR / "gen_calls.json"


def _today() -> str:
    """Return today's date as an ISO string (YYYY-MM-DD) in local time."""
    return datetime.date.today().isoformat()


def _read_counter(path: Path) -> dict:
    """Read a counter file, returning {date, count}.  Resets on date change."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("date") == _today():
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"date": _today(), "count": 0}


def _write_counter(path: Path, data: dict) -> None:
    """Persist a counter dict to disk atomically (best-effort)."""
    path.write_text(json.dumps(data), encoding="utf-8")


def increment_judge_counter() -> int:
    """Increment and persist the daily judge-call counter.

    Returns:
        The new total count for today, *after* incrementing.
    Raises:
        RuntimeError: if the daily cap has already been reached.
    """
    data = _read_counter(_JUDGE_COUNTER_FILE)
    if data["count"] >= MAX_DAILY_JUDGE_CALLS:
        raise RuntimeError(
            f"[RLHF] Daily judge-call cap ({MAX_DAILY_JUDGE_CALLS}) reached. "
            "Resets at midnight.  Adjust MAX_DAILY_JUDGE_CALLS or wait."
        )
    data["count"] += 1
    _write_counter(_JUDGE_COUNTER_FILE, data)
    return data["count"]


def increment_generation_counter() -> int:
    """Increment and persist the daily generation-call counter.

    Returns:
        The new total count for today, *after* incrementing.
    Raises:
        RuntimeError: if the daily cap has already been reached.
    """
    data = _read_counter(_GEN_COUNTER_FILE)
    if data["count"] >= MAX_DAILY_GENERATION_CALLS:
        raise RuntimeError(
            f"[RLHF] Daily generation-call cap ({MAX_DAILY_GENERATION_CALLS}) reached. "
            "Resets at midnight.  Adjust MAX_DAILY_GENERATION_CALLS or wait."
        )
    data["count"] += 1
    _write_counter(_GEN_COUNTER_FILE, data)
    return data["count"]


def get_daily_counts() -> dict:
    """Return the current daily call counts for monitoring / debugging.

    Returns:
        dict with keys ``judge_calls`` and ``generation_calls``.
    """
    j = _read_counter(_JUDGE_COUNTER_FILE)
    g = _read_counter(_GEN_COUNTER_FILE)
    return {
        "judge_calls": j["count"],
        "generation_calls": g["count"],
        "date": _today(),
    }

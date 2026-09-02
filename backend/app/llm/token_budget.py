"""
backend/app/llm/token_budget.py

Token counting and cost budget enforcement for all LLM calls.

Tracks usage per (tenant_id, department, date) in SQLite and alerts
when a department exceeds its daily cost budget.

tiktoken is used when available (fast BPE tokenizer); falls back to
approximate char/4 counting when not installed or the model is unknown.
"""
from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Cost table: USD per 1000 tokens (input / output)
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b":               (0.003, 0.015),
    "llama3.2:1b":                        (0.0,   0.0),
    "llama3.2:3b":                        (0.0,   0.0),
    "default":                            (0.002, 0.010),
}

_lock = threading.Lock()


def count_tokens(text: str, model: str = "default") -> int:
    """Count tokens using tiktoken if available, else char/4 approximation."""
    try:
        import tiktoken
        enc_name = "cl100k_base"
        if "gpt-4o" in model or "gpt-4" in model:
            enc_name = "o200k_base"
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Estimate USD cost for a call given token counts and model name."""
    key = model.lower().replace("ollama/", "")
    in_cost, out_cost = _COST_PER_1K.get(key, _COST_PER_1K["default"])
    return round((prompt_tokens / 1000 * in_cost) + (completion_tokens / 1000 * out_cost), 6)


class TokenUsageStore:
    """Thread-safe SQLite store for per-department token usage tracking."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "controlplane.db"
            )
        self._db_path = os.path.abspath(db_path)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with _lock:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS llm_token_usage (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id         TEXT    NOT NULL DEFAULT 'default',
                        department        TEXT    NOT NULL DEFAULT 'default',
                        date              TEXT    NOT NULL,
                        model             TEXT    NOT NULL,
                        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens      INTEGER NOT NULL DEFAULT 0,
                        estimated_cost_usd REAL   NOT NULL DEFAULT 0.0,
                        created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_dept_date ON llm_token_usage(department, date)")
                conn.commit()
            finally:
                conn.close()

    def record_usage(
        self,
        *,
        tenant_id: str = "default",
        department: str = "default",
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Persist a usage record and return the estimated cost in USD."""
        total = prompt_tokens + completion_tokens
        cost = estimate_cost_usd(prompt_tokens, completion_tokens, model)
        date_str = datetime.date.today().isoformat()
        try:
            with _lock:
                conn = sqlite3.connect(self._db_path, check_same_thread=False)
                try:
                    conn.execute(
                        """INSERT INTO llm_token_usage
                             (tenant_id, department, date, model,
                              prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd)
                             VALUES (?,?,?,?,?,?,?,?)""",
                        (tenant_id, department, date_str, model,
                         prompt_tokens, completion_tokens, total, cost),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("TokenUsageStore.record_usage failed: %s", exc)
        return cost

    def daily_cost(self, department: str, date: Optional[str] = None) -> float:
        """Return total estimated cost (USD) for a department on a given date."""
        if date is None:
            date = datetime.date.today().isoformat()
        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) FROM llm_token_usage WHERE department=? AND date=?",
                    (department, date),
                ).fetchone()
                return float(row[0]) if row else 0.0
            finally:
                conn.close()
        except Exception:
            return 0.0

    def check_budget(
        self,
        department: str,
        daily_limit_usd: float = 50.0,
        tenant_id: str = "default",
    ) -> dict:
        """Check if the department is within its daily budget. Returns status dict."""
        spent = self.daily_cost(department)
        over_budget = spent >= daily_limit_usd
        if over_budget:
            logger.warning(
                "BUDGET ALERT: Department '%s' has spent $%.4f today (limit $%.2f)",
                department, spent, daily_limit_usd,
            )
        return {
            "department": department,
            "date": datetime.date.today().isoformat(),
            "spent_usd": round(spent, 6),
            "limit_usd": daily_limit_usd,
            "over_budget": over_budget,
            "remaining_usd": max(0.0, round(daily_limit_usd - spent, 6)),
        }


_store_instance: Optional[TokenUsageStore] = None


def get_usage_store() -> TokenUsageStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = TokenUsageStore()
    return _store_instance

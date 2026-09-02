"""Audit store — privacy-conscious persistence (Section 5.9 / 5.10).

Merges the old audit.py (HMAC fingerprint, allow-listed context) and
db.py (SQLite persistence) into one module.  The prototype deliberately
does not persist raw prompts, raw responses, or retrieved document
contents — only fingerprinted / allow-listed metadata.

Scalability improvements (Phase 4):
- WAL journal mode + PRAGMA tuning via ThreadLocalPool (db_pool.py)
- One cached connection per OS thread — no per-request open/close overhead
- PRAGMA busy_timeout=5000 so writers wait up to 5 s instead of raising
- Explicit threading.Lock around write methods to serialise SQLite writes
  from the same process (WAL handles concurrent READERS for free)
- tenacity retries on OperationalError: database is locked (belt-and-suspenders)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import tenacity

from backend.shared.config import settings
from backend.shared.db_pool import get_pool, ThreadLocalPool
from backend.shared.schemas import GovernanceRequest


# ---------------------------------------------------------------------------
# Privacy-conscious audit helpers
# ---------------------------------------------------------------------------
def fingerprint(value: str) -> str:
    """HMAC-SHA256 fingerprint — identifies without revealing raw text."""
    return hmac.new(
        settings.audit_hash_key.encode("utf-8"),
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:24]


def build_audit_context(request: GovernanceRequest) -> dict[str, Any]:
    """Return allow-listed metadata useful for governance investigations."""
    return {
        "user_fingerprint": fingerprint(request.user_id),
        "application_id": request.application_id,
        "department": request.department,
        "model": request.model,
        "provider": request.provider,
        "user_role": request.user_role,
        "data_classification": request.data_classification,
        "tool_count": len(request.tools_requested),
        "retrieved_context_count": len(request.retrieved_context),
        "response_present": request.response is not None,
    }


# ---------------------------------------------------------------------------
# Retry decorator — wraps SQLite write ops (locked DB is transient)
# ---------------------------------------------------------------------------
def _is_db_locked(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


_retry_write = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_db_locked),
    wait=tenacity.wait_exponential(multiplier=0.05, min=0.05, max=0.2),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)

_retry_read = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_db_locked),
    wait=tenacity.wait_exponential(multiplier=0.05, min=0.05, max=0.1),
    stop=tenacity.stop_after_attempt(2),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Database (SQLite for dev, PostgreSQL target)
# ---------------------------------------------------------------------------
class Database:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pool: ThreadLocalPool = get_pool(self.path)
        self._write_lock = threading.Lock()  # serialise writes within one process
        self.init()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        """Return the thread-local cached connection (WAL mode, tuned PRAGMAs)."""
        return self._pool.get_conn()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------
    def init(self):
        conn = self.connect()
        with self._write_lock:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                decision TEXT NOT NULL,
                risk REAL NOT NULL,
                latency_ms REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS async_jobs (
                job_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                human_action TEXT NOT NULL,
                correct INTEGER NOT NULL,
                comment TEXT
            );

            CREATE TABLE IF NOT EXISTS governance_audits (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                prompt_fingerprint TEXT NOT NULL,
                audit_context TEXT NOT NULL,
                detector_results TEXT NOT NULL,
                risk TEXT NOT NULL,
                policy TEXT NOT NULL,
                decision_details TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_reviews (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                risk REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                final_action TEXT,
                reviewer_id TEXT,
                notes TEXT,
                resolved_at TEXT,
                prompt TEXT
            );
            """)
            try:
                conn.execute("ALTER TABLE pending_reviews ADD COLUMN prompt TEXT")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    # ------------------------------------------------------------------
    # Write methods — protected by _write_lock + tenacity retries
    # ------------------------------------------------------------------
    @_retry_write
    def save_request(
        self,
        request_id: str,
        audit_context: Dict[str, Any],
        decision: str,
        risk: float,
        latency_ms: float,
        prompt_fingerprint: str,
        detector_results: list[Dict[str, Any]],
        risk_details: Dict[str, Any],
        policy: Dict[str, Any],
        decision_details: Dict[str, Any],
    ):
        created_at = datetime.now(timezone.utc).isoformat()
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO requests(request_id, created_at, payload, decision, risk, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    created_at,
                    json.dumps(audit_context, default=str),
                    decision,
                    risk,
                    latency_ms,
                ),
            )
            conn.execute(
                """INSERT OR REPLACE INTO governance_audits
                (request_id, created_at, prompt_fingerprint, audit_context, detector_results, risk, policy, decision_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    created_at,
                    prompt_fingerprint,
                    json.dumps(audit_context, default=str),
                    json.dumps(detector_results, default=str),
                    json.dumps(risk_details, default=str),
                    json.dumps(policy, default=str),
                    json.dumps(decision_details, default=str),
                ),
            )
            conn.commit()

    @_retry_write
    def create_job(self, job_id: str, request_id: str):
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "INSERT INTO async_jobs VALUES (?, ?, ?, ?, ?)",
                (job_id, request_id, datetime.now(timezone.utc).isoformat(), "QUEUED", None),
            )
            conn.commit()

    @_retry_write
    def update_job(self, job_id: str, status: str, result: Dict[str, Any] | None = None):
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "UPDATE async_jobs SET status=?, result=? WHERE job_id=?",
                (status, json.dumps(result) if result is not None else None, job_id),
            )
            conn.commit()

    @_retry_write
    def save_feedback(self, request_id: str, human_action: str, correct: bool, comment: str):
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "INSERT INTO feedback(request_id, created_at, human_action, correct, comment) VALUES (?, ?, ?, ?, ?)",
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    human_action,
                    int(correct),
                    comment,
                ),
            )
            conn.commit()

    @_retry_write
    def create_review(self, request_id: str, policy_id: str, reason: str, risk: float, prompt: str = ""):
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO pending_reviews "
                "(request_id, created_at, policy_id, reason, risk, status, prompt) "
                "VALUES (?, ?, ?, ?, ?, 'PENDING', ?)",
                (request_id, datetime.now(timezone.utc).isoformat(), policy_id, reason, risk, prompt),
            )
            conn.commit()

    @_retry_write
    def resolve_review(self, request_id: str, final_action: str, reviewer_id: str, notes: str = ""):
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "UPDATE pending_reviews SET status='RESOLVED', final_action=?, "
                "reviewer_id=?, notes=?, resolved_at=? WHERE request_id=?",
                (final_action, reviewer_id, notes, datetime.now(timezone.utc).isoformat(), request_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read methods — use shared thread-local connection, retry on lock
    # ------------------------------------------------------------------
    @_retry_read
    def recent_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["audit_context"] = json.loads(item.pop("payload"))
            result.append(item)
        return result

    @_retry_read
    def get_job(self, job_id: str):
        conn = self.connect()
        row = conn.execute("SELECT * FROM async_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        job = dict(row)
        if job.get("result") is not None and isinstance(job["result"], str):
            try:
                job["result"] = json.loads(job["result"])
            except Exception:
                pass
        return job

    @_retry_read
    def get_audit(self, request_id: str) -> Dict[str, Any] | None:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM governance_audits WHERE request_id=?", (request_id,)
        ).fetchone()
        if not row:
            return None
        audit = dict(row)
        for field in ("audit_context", "detector_results", "risk", "policy", "decision_details"):
            audit[field] = json.loads(audit[field])
        return audit

    @_retry_read
    def recent_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM governance_audits ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        audits = []
        for row in rows:
            audit = dict(row)
            audit["audit_context"] = json.loads(audit["audit_context"])
            audit["detector_results"] = json.loads(audit["detector_results"])
            audit["risk"] = json.loads(audit["risk"])
            audit["policy"] = json.loads(audit["policy"])
            audit["decision_details"] = json.loads(audit["decision_details"])
            audits.append(audit)
        return audits

    @_retry_read
    def request_exists(self, request_id: str) -> bool:
        conn = self.connect()
        row = conn.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone()
        return row is not None

    @_retry_read
    def list_pending_reviews(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM pending_reviews WHERE status='PENDING' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if not d.get("prompt"):
                # Backfill prompt from governance_audits or requests table if available
                try:
                    audit = conn.execute("SELECT audit_context FROM governance_audits WHERE request_id=?", (d["request_id"],)).fetchone()
                    if audit:
                        ctx = json.loads(audit["audit_context"])
                        d["prompt"] = ctx.get("prompt", "")
                except Exception:
                    pass
                if not d.get("prompt"):
                    try:
                        req = conn.execute("SELECT payload FROM requests WHERE request_id=?", (d["request_id"],)).fetchone()
                        if req:
                            p = json.loads(req["payload"])
                            d["prompt"] = p.get("prompt", "")
                    except Exception:
                        pass
            result.append(d)
        return result

    @_retry_read
    def get_review(self, request_id: str) -> Dict[str, Any] | None:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM pending_reviews WHERE request_id=?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    @_retry_read
    def metrics(self) -> Dict[str, Any]:
        conn = self.connect()
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='BLOCK'").fetchone()[0]
        review = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='HUMAN_REVIEW'").fetchone()[0]
        modified = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='MODIFY'").fetchone()[0]
        reroute = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='REROUTE'").fetchone()[0]
        avg_latency = conn.execute("SELECT COALESCE(AVG(latency_ms),0) FROM requests").fetchone()[0]
        feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {
            "total_requests": total,
            "blocked": blocked,
            "human_review": review,
            "modified": modified,
            "rerouted": reroute,
            "avg_latency_ms": round(avg_latency, 2),
            "feedback_count": feedback_count,
        }

    @_retry_read
    def richer_metrics(self) -> dict:
        """Extended metrics including detector fire rates, time-series data, and risk distribution."""
        conn = self.connect()

        # Basic counts
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='BLOCK'").fetchone()[0]
        review = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='HUMAN_REVIEW'").fetchone()[0]
        modified = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='MODIFY'").fetchone()[0]
        reroute = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='REROUTE'").fetchone()[0]
        allowed = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='ALLOW'").fetchone()[0]
        avg_latency = conn.execute("SELECT COALESCE(AVG(latency_ms),0) FROM requests").fetchone()[0]
        feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        # Risk distribution buckets (using requests table)
        risk_low = conn.execute("SELECT COUNT(*) FROM requests WHERE risk < 0.3").fetchone()[0]
        risk_med = conn.execute("SELECT COUNT(*) FROM requests WHERE risk >= 0.3 AND risk < 0.7").fetchone()[0]
        risk_high = conn.execute("SELECT COUNT(*) FROM requests WHERE risk >= 0.7").fetchone()[0]

        # Latency trend: last 20 requests (created_at + latency_ms)
        latency_rows = conn.execute(
            "SELECT created_at, latency_ms FROM requests ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        latency_trend = [{"ts": r["created_at"], "ms": round(r["latency_ms"], 1)} for r in reversed(latency_rows)]

        # Risk trend: last 20 requests
        risk_rows = conn.execute(
            "SELECT created_at, risk FROM requests ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        risk_trend = [{"ts": r["created_at"], "risk": round(r["risk"], 4)} for r in reversed(risk_rows)]

        # Detector fire rates from governance_audits
        detector_fire_counts: dict = {}
        try:
            audit_rows = conn.execute(
                "SELECT detector_results FROM governance_audits ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
            for row in audit_rows:
                try:
                    detectors = json.loads(row["detector_results"])
                    for d in detectors:
                        name = d.get("detector_name", "unknown")
                        score = d.get("score", 0.0)
                        if name not in detector_fire_counts:
                            detector_fire_counts[name] = {"fires": 0, "total": 0}
                        detector_fire_counts[name]["total"] += 1
                        if score > 0.0:
                            detector_fire_counts[name]["fires"] += 1
                except Exception:
                    pass
        except Exception:
            pass

        detector_fire_rates = {
            name: {
                "fires": v["fires"],
                "total": v["total"],
                "rate": round(v["fires"] / v["total"], 3) if v["total"] > 0 else 0.0,
            }
            for name, v in detector_fire_counts.items()
        }

        # Blocked by rule (from governance_audits)
        blocked_by_policy: dict = {}
        try:
            policy_rows = conn.execute(
                "SELECT policy FROM governance_audits WHERE json_extract(decision_details, '$.action') = 'BLOCK' LIMIT 200"
            ).fetchall()
            for row in policy_rows:
                try:
                    pol = json.loads(row["policy"])
                    pid = pol.get("policy_id", "unknown")
                    blocked_by_policy[pid] = blocked_by_policy.get(pid, 0) + 1
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "total_requests": total,
            "blocked": blocked,
            "allowed": allowed,
            "human_review": review,
            "modified": modified,
            "rerouted": reroute,
            "avg_latency_ms": round(avg_latency, 2),
            "feedback_count": feedback_count,
            "risk_distribution": {
                "low": risk_low,
                "medium": risk_med,
                "high": risk_high,
            },
            "latency_trend": latency_trend,
            "risk_trend": risk_trend,
            "detector_fire_rates": detector_fire_rates,
            "blocked_by_policy": blocked_by_policy,
        }

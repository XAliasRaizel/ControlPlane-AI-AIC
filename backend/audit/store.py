"""Audit store — privacy-conscious persistence (Section 5.9 / 5.10).

Merges the old audit.py (HMAC fingerprint, allow-listed context) and
db.py (SQLite persistence) into one module.  The prototype deliberately
does not persist raw prompts, raw responses, or retrieved document
contents — only fingerprinted / allow-listed metadata.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.shared.config import settings
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
# Database (SQLite for dev, PostgreSQL target)
# ---------------------------------------------------------------------------
class Database:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self):
        conn = self.connect()
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
        """)
        conn.commit()
        conn.close()

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
        conn.close()

    def create_job(self, job_id: str, request_id: str):
        conn = self.connect()
        conn.execute(
            "INSERT INTO async_jobs VALUES (?, ?, ?, ?, ?)",
            (job_id, request_id, datetime.now(timezone.utc).isoformat(), "QUEUED", None),
        )
        conn.commit()
        conn.close()

    def update_job(self, job_id: str, status: str, result: Dict[str, Any] | None = None):
        conn = self.connect()
        conn.execute(
            "UPDATE async_jobs SET status=?, result=? WHERE job_id=?",
            (status, json.dumps(result) if result is not None else None, job_id),
        )
        conn.commit()
        conn.close()

    def save_feedback(self, request_id: str, human_action: str, correct: bool, comment: str):
        conn = self.connect()
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
        conn.close()

    def recent_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        result = []
        for row in rows:
            item = dict(row)
            item["audit_context"] = json.loads(item.pop("payload"))
            result.append(item)
        return result

    def get_job(self, job_id: str):
        conn = self.connect()
        row = conn.execute("SELECT * FROM async_jobs WHERE job_id=?", (job_id,)).fetchone()
        conn.close()
        if not row:
            return None
        job = dict(row)
        if job["result"] is not None:
            job["result"] = json.loads(job["result"])
        return job

    def get_audit(self, request_id: str) -> Dict[str, Any] | None:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM governance_audits WHERE request_id=?", (request_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        audit = dict(row)
        for field in ("audit_context", "detector_results", "risk", "policy", "decision_details"):
            audit[field] = json.loads(audit[field])
        return audit

    def recent_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM governance_audits ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        audits = []
        for row in rows:
            audit = dict(row)
            for field in ("audit_context", "detector_results", "risk", "policy", "decision_details"):
                audit[field] = json.loads(audit[field])
            audits.append(audit)
        return audits

    def request_exists(self, request_id: str) -> bool:
        conn = self.connect()
        row = conn.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone()
        conn.close()
        return row is not None

    def metrics(self) -> Dict[str, Any]:
        conn = self.connect()
        total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='BLOCK'").fetchone()[0]
        review = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='HUMAN_REVIEW'").fetchone()[0]
        modified = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='MODIFY'").fetchone()[0]
        reroute = conn.execute("SELECT COUNT(*) FROM requests WHERE decision='REROUTE'").fetchone()[0]
        avg_latency = conn.execute("SELECT COALESCE(AVG(latency_ms),0) FROM requests").fetchone()[0]
        feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        return {
            "total_requests": total,
            "blocked": blocked,
            "human_review": review,
            "modified": modified,
            "rerouted": reroute,
            "avg_latency_ms": round(avg_latency, 2),
            "feedback_count": feedback_count,
        }

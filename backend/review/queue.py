"""Human review queue (Section 5.7).

FIX: this used to unconditionally downgrade every HUMAN_REVIEW decision to
BLOCK ("Phase-1 fallback, no review UI yet") -- which was the *correct*
call when it was written, because the risk engine's dilution bug (see
risk/engine.py) meant HUMAN_REVIEW was already close to unreachable via the
risk-threshold rules, and finance.yaml's data-classification-triggered
route to it had no real destination to land in anyway. Now that both are
fixed, a HUMAN_REVIEW decision is held as a genuinely pending review --
persisted so it survives a restart, and resolvable via resolve() -- instead
of being silently collapsed into BLOCK before anyone ever sees it.
"""

from __future__ import annotations

import logging

from backend.shared.config import settings
from backend.shared.schemas import GovernanceDecision

logger = logging.getLogger("controlplane.review")


class ReviewQueue:
    def __init__(self, db=None):
        # Local import avoids a circular import with audit.store at module load time.
        if db is None:
            from backend.audit.store import Database
            db = Database(settings.db_path)
        self.db = db

    def enqueue(self, decision: GovernanceDecision) -> GovernanceDecision:
        """Persist a HUMAN_REVIEW decision as pending. Returns it unchanged --
        the queue's job is to hold the decision, not to make a different one."""
        if decision.action != "HUMAN_REVIEW":
            return decision

        self.db.create_review(
            request_id=decision.request_id,
            policy_id=decision.policy_id or "unknown",
            reason=decision.reason,
            risk=decision.risk_snapshot.overall_risk,
        )
        logger.info("HUMAN_REVIEW queued for request %s", decision.request_id)
        return decision

    def resolve(self, request_id: str, final_action: str, reviewer_id: str, notes: str = "") -> dict:
        self.db.resolve_review(request_id, final_action, reviewer_id, notes)
        logger.info(
            "Review %s resolved by %s -> %s", request_id, reviewer_id, final_action
        )
        return self.db.get_review(request_id)

    def list_pending(self, limit: int = 50) -> list[dict]:
        return self.db.list_pending_reviews(limit)

    def pending_count(self) -> int:
        return len(self.db.list_pending_reviews(limit=10_000))

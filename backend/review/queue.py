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

    def enqueue(self, decision: GovernanceDecision, prompt: str = "") -> GovernanceDecision:
        """Persist a decision as pending review. Returns it unchanged --
        the queue holds the decision so human reviewers can inspect, audit,
        and override decisions in real-time in Tab 5."""
        self.db.create_review(
            request_id=decision.request_id,
            policy_id=decision.policy_id or "default-allow",
            reason=decision.reason or "Governance evaluation",
            risk=decision.risk_snapshot.overall_risk if decision.risk_snapshot else 0.0,
            prompt=prompt,
        )
        logger.info("Decision queued for Human Review for request %s (action=%s)", decision.request_id, decision.action)
        return decision

    def resolve(self, request_id: str, final_action: str, reviewer_id: str, notes: str = "",
                request=None, hot_path_results=None) -> dict:
        """Resolve a pending HUMAN_REVIEW decision.

        Args:
            request_id: The governance request ID.
            final_action: "ALLOW" or "BLOCK" — the human reviewer's decision.
            reviewer_id: Identifier of the reviewer.
            notes: Optional reviewer notes.
            request: Optional original GovernanceRequest for training signal collection.
            hot_path_results: Optional list[DetectorResult] for training signal collection.
        """
        self.db.resolve_review(request_id, final_action, reviewer_id, notes)
        logger.info(
            "Review %s resolved by %s -> %s", request_id, reviewer_id, final_action
        )

        # Capture gold-label training signal from human review
        if request is not None:
            try:
                from backend.async_pipeline.training_signal_collector import collect_human_override
                collect_human_override(
                    request=request,
                    hot_path_results=hot_path_results or [],
                    final_action=final_action,
                    reviewer_id=reviewer_id,
                )
            except Exception as exc:
                logger.debug("Training signal collection (human) suppressed: %s", exc)

        return self.db.get_review(request_id)

    def list_pending(self, limit: int = 50) -> list[dict]:
        return self.db.list_pending_reviews(limit)

    def pending_count(self) -> int:
        return len(self.db.list_pending_reviews(limit=10_000))

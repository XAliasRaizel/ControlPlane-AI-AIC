"""Human review queue (Section 5.7).

Until the review UI exists (Phase 5), HUMAN_REVIEW decisions are
downgraded to BLOCK-and-log so the golden path works without a UI
dependency.
"""

import logging

from backend.shared.schemas import GovernanceDecision

logger = logging.getLogger("controlplane.review")


class ReviewQueue:
    """In-memory stub.  Phase 5 will add persistence and a web UI."""

    def __init__(self):
        self._pending: dict[str, GovernanceDecision] = {}

    def enqueue(self, decision: GovernanceDecision) -> GovernanceDecision:
        """Accept a HUMAN_REVIEW decision; for now, downgrade to BLOCK."""
        if decision.action != "HUMAN_REVIEW":
            return decision
        logger.info(
            "HUMAN_REVIEW downgraded to BLOCK for request %s (no review UI yet)",
            decision.request_id,
        )
        return decision.model_copy(
            update={
                "action": "BLOCK",
                "reason": f"[auto-downgraded from HUMAN_REVIEW] {decision.reason}",
            }
        )

    def pending_count(self) -> int:
        return len(self._pending)

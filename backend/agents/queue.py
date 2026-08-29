"""
backend/app/agents/queue.py

A minimal in-memory human-review queue for tool calls the governor
routed to HUMAN_REVIEW. Swap the dict-backed store for your existing
review-queue / audit persistence (SQLite or Postgres) when you wire
this into the rest of ControlPlane -- the interface (`enqueue`,
`list_pending`, `resolve`) is deliberately small so that swap is a
one-file change.
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from .models import GovernanceDecision, PendingToolCall, ToolCallContext, ToolRiskSignal


class ReviewQueue:
    def __init__(self) -> None:
        self._pending: Dict[str, PendingToolCall] = {}

    def enqueue(self, ctx: ToolCallContext, risk: ToolRiskSignal, decision: GovernanceDecision) -> PendingToolCall:
        pending_id = f"pend_{uuid.uuid4().hex[:8]}"
        record = PendingToolCall(pending_id=pending_id, context=ctx, risk_signal=risk, decision=decision)
        self._pending[pending_id] = record
        return record

    def list_pending(self) -> List[PendingToolCall]:
        return [p for p in self._pending.values() if p.status == "PENDING"]

    def get(self, pending_id: str) -> Optional[PendingToolCall]:
        return self._pending.get(pending_id)

    def resolve(self, pending_id: str, approve: bool, resolved_by: str) -> Optional[PendingToolCall]:
        record = self._pending.get(pending_id)
        if record is None or record.status != "PENDING":
            return None
        record.status = "APPROVED" if approve else "REJECTED"
        record.resolved_by = resolved_by
        record.resolved_at = time.time()
        return record

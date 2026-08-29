"""
backend/app/agents/models.py

Plain dataclasses -- zero hard dependency on Pydantic, so this module
can be unit-tested anywhere Python runs, including environments where
FastAPI/Pydantic aren't installed yet. When you wire this into the
real API gateway, wrap these in Pydantic BaseModels at the HTTP
boundary only (see router.py); keep the dataclasses as the internal
contract so the governance core stays framework-agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallContext:
    """Everything the governor needs to know about *who* is asking and *where*."""
    session_id: str
    role: str                      # e.g. "employee", "manager", "admin", "intern"
    application: str               # e.g. "support-agent", "finance-agent"
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    session_risk: float = 0.0      # carried over from the hot-path risk engine / session tracker
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolRiskSignal:
    """Mirrors the DetectorResult shape used elsewhere in ControlPlane
    (score + confidence + evidence) so this reads like the same
    system, not a bolted-on side project."""
    tool: str
    score: float
    confidence: float
    factors: Dict[str, Any] = field(default_factory=dict)
    raw_context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class GovernanceDecision:
    action: str                    # "ALLOW" | "BLOCK" | "HUMAN_REVIEW"
    reason: str
    risk: float
    matched_rule: Optional[str] = None
    confidence: float = 1.0


@dataclass
class PendingToolCall:
    pending_id: str
    context: ToolCallContext
    risk_signal: ToolRiskSignal
    decision: GovernanceDecision
    created_at: float = field(default_factory=time.time)
    status: str = "PENDING"        # PENDING | APPROVED | REJECTED
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None

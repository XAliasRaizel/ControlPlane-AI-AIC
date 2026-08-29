"""ControlPlane.ai — Canonical Data Contracts (Section 6).

This is the single source of truth for every data shape in the system.
Every component reads and writes these schemas and nothing else.
Implement this file first, before any component logic — everything else
imports from it.

Validated against pydantic 2.x.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Governance actions — used throughout the pipeline
# ---------------------------------------------------------------------------
DecisionAction = Literal["ALLOW", "MODIFY", "REROUTE", "HUMAN_REVIEW", "BLOCK"]


# ---------------------------------------------------------------------------
# 1. GovernanceRequest — canonical envelope for one AI interaction
# ---------------------------------------------------------------------------
class GovernanceRequest(BaseModel):
    """Canonical envelope for one AI interaction entering ControlPlane."""

    request_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str
    user_role: str = "user"
    department: Optional[str] = None
    application_id: str
    model: str = "demo-llm"
    provider: str = "local"
    prompt: str
    response: Optional[str] = None
    tools_requested: list[str] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)
    data_classification: Optional[str] = None  # e.g. "PUBLIC" | "INTERNAL" | "HIGH"
    fast_lane_webhook: Optional[str] = None
    session_id: Optional[str] = None  # NEW (Phase 9) — session tracking key for the accumulator

    # --- Backward-compatibility aliases (old field names accepted on input) ---
    # These are NOT persisted; they populate the canonical fields above via
    # model_validator.  Kept so existing callers / tests don't break.
    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# 2. DetectorResult — one detector's opinion on one request
# ---------------------------------------------------------------------------
class DetectorResult(BaseModel):
    """One detector's opinion on one request."""

    detector_name: str
    score: float  # 0.0-1.0, higher = riskier
    label: str  # e.g. "PII_DETECTED", "CLEAN"
    confidence: float  # 0.0-1.0
    evidence: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# 3. RiskAssessment — fused output of all detectors + context
# ---------------------------------------------------------------------------
class RiskAssessment(BaseModel):
    """Fused output of all detectors + context, produced by the Risk Engine."""

    request_id: str
    detector_results: list[DetectorResult]
    contextual_factors: dict[str, Any] = Field(default_factory=dict)
    dimensions: dict[str, float] = Field(default_factory=dict)  # {"privacy": 0.9, ...}
    overall_risk: float
    confidence: float
    session_risk: Optional[float] = None   # NEW (Phase 9) — max(EWMA, peak); None when accumulator disabled
    session_band: Optional[int] = None     # NEW (Phase 9) — 1/2/3 (band 4 = existing CRITICAL path, unchanged)


# ---------------------------------------------------------------------------
# 4. PolicyMatch — which policy rule matched
# ---------------------------------------------------------------------------
class PolicyMatch(BaseModel):
    policy_id: str
    policy_name: str
    matched_condition: str
    recommended_action: DecisionAction


# ---------------------------------------------------------------------------
# 5. GovernanceDecision — the final, auditable output of the pipeline
# ---------------------------------------------------------------------------
class GovernanceDecision(BaseModel):
    """The final, auditable output of the pipeline."""

    request_id: str
    action: DecisionAction
    reason: str
    policy_id: Optional[str] = None
    risk_snapshot: RiskAssessment
    modified_response: Optional[str] = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 6. HumanReviewOutcome — what a human reviewer decided
# ---------------------------------------------------------------------------
class HumanReviewOutcome(BaseModel):
    request_id: str
    reviewer_id: str
    original_action: str
    final_action: str
    notes: Optional[str] = None
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 7. AuditRecord — what gets persisted for every single request
# ---------------------------------------------------------------------------
class AuditRecord(BaseModel):
    """What gets persisted for every single request."""

    request: GovernanceRequest
    risk: RiskAssessment
    decision: GovernanceDecision
    human_outcome: Optional[HumanReviewOutcome] = None


# ---------------------------------------------------------------------------
# 8. API response model (needed for FastAPI, not in the spec schemas)
# ---------------------------------------------------------------------------
class GovernanceResponse(BaseModel):
    """Envelope returned by the /v1/govern endpoint."""

    request_id: str
    decision: GovernanceDecision
    risk: RiskAssessment
    detectors: list[DetectorResult]
    policy: PolicyMatch
    sanitized_response: Optional[str] = None
    async_job_id: Optional[str] = None
    policy_evidence: Optional[dict[str, Any]] = None
    fast_lane_pending: bool = False
    session_risk: Optional[float] = None   # NEW (Phase 9) — surfaced on response for demo visibility
    session_band: Optional[int] = None     # NEW (Phase 9)
    latency_ms: float


# ---------------------------------------------------------------------------
# 9. Supporting models for other endpoints
# ---------------------------------------------------------------------------
class PolicySummary(BaseModel):
    policy_name: str
    policy_version: str
    rules: list[dict[str, Any]]


class FeedbackRequest(BaseModel):
    request_id: str
    reviewer_id: str = "anonymous"
    original_action: DecisionAction
    final_action: DecisionAction
    notes: str = ""

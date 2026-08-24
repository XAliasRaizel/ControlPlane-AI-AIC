from datetime import datetime, timezone
import re
from backend.shared.schemas import GovernanceRequest, RiskAssessment, PolicyMatch, GovernanceDecision

def make_decision(request: GovernanceRequest, risk: RiskAssessment, policy: PolicyMatch) -> GovernanceDecision:
    action = policy.recommended_action or "ALLOW"
    reason = f"Matched policy rule: {policy.policy_id} ({policy.matched_condition})"
    
    if action == "HUMAN_REVIEW":
        action = "BLOCK"
        reason += " - Downgraded to BLOCK per Phase-1 fallback."
        
    confidence = min(0.99, 0.60 + abs(risk.overall_risk - 0.5) * 0.8)
    
    return GovernanceDecision(
        request_id=request.request_id,
        action=action,
        reason=reason,
        policy_id=policy.policy_id,
        risk_snapshot=risk,
        decided_at=datetime.now(timezone.utc)
    )

def sanitize_response(response: str | None, decision: GovernanceDecision) -> str | None:
    if response is None:
        return None
    if decision.action in {"BLOCK", "HUMAN_REVIEW"}:
        return None
    if decision.action != "MODIFY":
        return response

    masked = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL_REDACTED]",
        response,
    )
    masked = re.sub(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b", "[PHONE_REDACTED]", masked)
    masked = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", masked)
    masked = re.sub(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b", "[GOVERNMENT_ID_REDACTED]", masked)
    return masked + "\n\n[ControlPlane.ai: response sanitized by policy.]"

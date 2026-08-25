import re

from backend.shared.schemas import GovernanceRequest, RiskAssessment, PolicyMatch, GovernanceDecision

def make_decision(request: GovernanceRequest, risk: RiskAssessment, policy: PolicyMatch) -> GovernanceDecision:
    # FIX: the HUMAN_REVIEW -> BLOCK downgrade used to happen here, which
    # meant make_decision() could never honestly report "this needs a human"
    # -- by the time it returned, it had already decided unilaterally. That
    # downgrade is still the correct Phase-1 *fallback*, but it belongs in
    # the review queue (Section 5.7), which is the component that actually
    # knows whether a real queue exists yet to hold the decision instead.
    action = policy.recommended_action or "ALLOW"
    reason = f"Matched policy rule: {policy.policy_id} ({policy.matched_condition})"

    # FIX: `confidence` was computed here and then never used -- GovernanceDecision
    # has no such field, so this was dead code. Decision-level confidence is
    # already captured properly on risk.confidence (RiskAssessment); removed
    # rather than silently discarded.

    return GovernanceDecision(
        request_id=request.request_id,
        action=action,
        reason=reason,
        policy_id=policy.policy_id,
        risk_snapshot=risk,
    )


def sanitize_response(response: str | None, decision: GovernanceDecision) -> str | None:
    if decision.action == "BLOCK":
        return None
    if decision.action == "MODIFY" and response:
        cleaned = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "[EMAIL_REDACTED]", response,
        )
        cleaned = re.sub(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b", "[PHONE_REDACTED]", cleaned)
        cleaned = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", cleaned)
        cleaned = re.sub(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b", "[AADHAAR_REDACTED]", cleaned)
        return cleaned + "\n\n[ControlPlane.ai: response sanitized by policy.]"
    return response

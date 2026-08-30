import re

from backend.shared.schemas import GovernanceRequest, RiskAssessment, PolicyMatch, GovernanceDecision

# A lightweight "this response probably contains something worth checking"
# signal. Not exhaustive -- just enough to distinguish "there's likely
# sensitive-shaped content our specific patterns should have caught" from
# "this response is just generic text with nothing to redact in the first
# place." Only the former should escalate to BLOCK when redaction has no
# effect; escalating the latter would incorrectly block harmless fallback
# responses that happen to share a MODIFY decision with a prompt that merely
# mentioned a sensitive topic (the decision is made on the prompt, before
# the response exists).
_SENSITIVE_SHAPE_HINTS = re.compile(r"\$\d|@|\bAccount\b|\bID:|\bending in\b|\d{3,}", re.I)


def make_decision(request: GovernanceRequest, risk: RiskAssessment, policy: PolicyMatch) -> GovernanceDecision:
    action = policy.recommended_action or "ALLOW"
    reason = f"Matched policy rule: {policy.policy_id} ({policy.matched_condition})"

    # Abstention Path (Low Confidence Routing)
    # Only triggers when a detector found potential risk (score > 0.0) but is uncertain (confidence < 0.70)
    low_conf_hits = [
        d for d in risk.detector_results
        if d.score > 0.0 and d.confidence < 0.70 and d.label not in (
            "CLEAN", "LOW", "NOT_APPLICABLE", "OPTIMAL", "COMPLIANT", "claims_grounded", "no_checkable_claims"
        )
    ]
    if low_conf_hits and action not in ("HUMAN_REVIEW", "BLOCK"):
        action = "HUMAN_REVIEW"
        hit_names = ", ".join(d.detector_name for d in low_conf_hits)
        reason = f"Abstention path triggered due to low confidence (<0.70) on: {hit_names}"


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
    if decision.action == "HUMAN_REVIEW":
        return None
    if decision.action == "MODIFY" and response:
        cleaned = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "[EMAIL_REDACTED]", response,
        )
        cleaned = re.sub(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b", "[PHONE_REDACTED]", cleaned)
        cleaned = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", cleaned)
        cleaned = re.sub(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b", "[AADHAAR_REDACTED]", cleaned)
        # FIX: a live run surfaced a response claiming "[sanitized by policy]"
        # while "Account ID: ACC-2024-0847" and "Bank account ending in 4521"
        # were still fully visible -- neither matched any pattern above.
        cleaned = re.sub(r"\b[A-Z]{2,5}-\d{4}-\d{3,6}\b", "[ACCOUNT_ID_REDACTED]", cleaned)
        cleaned = re.sub(r"\bending in \d{3,6}\b", "ending in [REDACTED]", cleaned, flags=re.I)
        cleaned = re.sub(r"\$[\d,]+(?:\.\d{2})?\b", "[AMOUNT_REDACTED]", cleaned)

        if cleaned == response:
            # Nothing matched. If the response looks sensitive-shaped anyway
            # (account/ID markers, amounts, long digit runs), redaction
            # silently failing is worse than blocking -- it would ship a
            # "[sanitized by policy]" label on a response that leaked
            # everything. Escalate instead. A genuinely generic response
            # (nothing sensitive to begin with) passes through unchanged,
            # unescalated, and without a misleading "sanitized" label.
            if _SENSITIVE_SHAPE_HINTS.search(response):
                return None
            return response

        return cleaned + "\n\n[ControlPlane.ai: response sanitized by policy.]"
    return response

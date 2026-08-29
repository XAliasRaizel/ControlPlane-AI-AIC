"""
backend/detectors/authorization.py -- Deterministic RBAC access check (hot-path).

Uses the shared sensitive_terms module as its single source of truth for
keyword-to-permission mappings, so coverage cannot diverge from pii.py.
"""

from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.shared.sensitive_terms import (
    find_keyword_hits,
    CATEGORY_PERMISSION,
    check_safety_net,
)
from backend.detectors.base import BaseDetector, register
from backend.shared.model_backend import consult_sensitive_intent


@register
class AuthorizationDetector(BaseDetector):
    name = "authorization"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = request.prompt.lower()
        auth_context = context.get("auth_context", {})

        # Find which sensitive categories are mentioned
        keyword_hits = find_keyword_hits(text)

        denied_resources = []
        for _kw, cat_name in keyword_hits:
            permission = CATEGORY_PERMISSION.get(cat_name, "")
            if permission and not bool(auth_context.get(permission, False)):
                denied_resources.append(cat_name)

        # Fail-cautious: if safety net triggers but no explicit keyword
        # matched, and the user doesn't have broad access, flag it
        if not denied_resources and not keyword_hits:
            safety_triggered, _, _ = check_safety_net(text)
            if safety_triggered:
                # Check if user has ANY access permissions
                has_any_access = any(auth_context.get(perm, False)
                                     for perm in auth_context)
                if not has_any_access:
                    denied_resources.append("unrecognized_sensitive_data")

        # Semantic Override (Phase 10):
        # If we have keyword matches but the semantic matcher says this is an aggregate/general query
        # (margin < threshold -> fires=False), suppress the keyword match.
        if denied_resources:
            intent_result = consult_sensitive_intent(request.prompt)
            if intent_result is not None:
                _, fires = intent_result
                if not fires:
                    # Semantic matcher determined this is NOT a targeted request
                    denied_resources = []

        score = 1.0 if denied_resources else 0.0
        confidence = 0.99 if denied_resources else 0.98
        label = "DENIED" if denied_resources else "AUTHORIZED"
        evidence = [f"unauthorized:{r}" for r in denied_resources]

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=evidence,
        )

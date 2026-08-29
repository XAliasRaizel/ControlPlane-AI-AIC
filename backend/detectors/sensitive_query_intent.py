from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register
from backend.shared.model_backend import consult_sensitive_intent

@register
class SensitiveQueryIntentDetector(BaseDetector):
    name = "sensitive_query_intent"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = request.prompt

        # Gated by model backend; if unavailable (env var unset), returns None
        intent_result = consult_sensitive_intent(text)

        if intent_result is None:
            # Additive fallback: if semantic model is not running, this detector is a no-op
            # and leaves the authorization/pii detectors to handle everything (Phase 9 behavior).
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                label="CLEAN",
                confidence=0.99,
                evidence=["model_unavailable_or_unset"]
            )

        margin, fires = intent_result

        # This detector specifically catches the 8 missed sensitive prompts that have NO
        # keyword match in authorization/pii (e.g., "what does she make?").
        # If the semantic model fires (margin >= threshold), we flag it.
        
        if fires:
            # We cap the score at 0.65 to mirror the "ambiguous/implicit request" level
            # from PII detector, since a semantic match on "tell me his SSN" is still
            # just a request for data, not a confirmed data leak.
            # (Though if they are asking about SSN, it's pretty risky!)
            # We'll use 0.65 to ensure it escalates if combined with other signals,
            # but doesn't instantly block on its own without context.
            score = 0.65
            label = "SENSITIVE_INTENT_DETECTED"
            confidence = 0.85
        else:
            score = 0.0
            label = "CLEAN"
            confidence = 0.90

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=[f"margin={margin:.4f}"]
        )

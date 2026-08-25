import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register

# FIX: every hit used to feed into one `score = 0.45 + 0.2*(n-1)` formula and
# one hardcoded `confidence = 0.94`, regardless of whether the text actually
# contained a PII *value* (an email, a phone number) or merely a *request*
# about a sensitive topic ("show me the salary details"). Those are
# genuinely different confidence levels -- a confirmed value is strong
# evidence; a bare mention of "salary" is a weaker, more ambiguous signal on
# its own (the real violation in that case is usually the authorization
# check, not the PII detector). Splitting them also means this detector can
# finally express something other than "0.0, very sure" or "~0.85+, very
# sure" -- a precondition for HUMAN_REVIEW's "low confidence" trigger ever
# being reachable for a PII-flavored case.
_VALUE_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(?:\+91[- ]?)?[6-9]\d{9}\b",
    "card": r"\b(?:\d[ -]*?){13,19}\b",
    "aadhaar_like": r"\b\d{4}[ -]\d{4}[ -]\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "api_key": r"\b(?:sk|api)[_-][A-Za-z0-9_-]{12,}\b",
}
_REQUEST_PATTERNS = {
    "phone_request": r"\b(?:phone\s*number|mobile\s*number|contact\s*number|phone|mobile)\b",
    "salary_request": r"\b(?:salary|compensation|pay|payroll|wage|ctc)\b",
    "account_request": r"\b(?:account\s*access|account\s*details|account\s*info|bank\s*account|account\s*number)\b",
    "personal_data_request": r"\b(?:personal\s*phone|personal\s*email|private\s*data|personal\s*info|personal\s*details)\b",
}


@register
class PIIDetector(BaseDetector):
    name = "pii"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}"

        value_hits = [label for label, p in _VALUE_PATTERNS.items() if re.search(p, text, re.I)]
        request_hits = [label for label, p in _REQUEST_PATTERNS.items() if re.search(p, text, re.I)]

        if value_hits:
            # A confirmed PII value is present -- high score, high confidence,
            # boosted slightly if the prompt was also *asking about* the topic.
            score = min(1.0, 0.7 + 0.1 * (len(value_hits) - 1) + (0.1 if request_hits else 0.0))
            confidence = min(0.97, 0.90 + 0.02 * len(value_hits))
            label = "PII_DETECTED"
        elif request_hits:
            # Only a request/mention of a sensitive topic, no confirmed value
            # yet -- real signal, but genuinely more ambiguous.
            score = min(0.65, 0.35 + 0.12 * (len(request_hits) - 1))
            confidence = min(0.75, 0.55 + 0.05 * len(request_hits))
            label = "PII_REQUEST_AMBIGUOUS"
        else:
            score = 0.0
            confidence = 0.90
            label = "CLEAN"

        return DetectorResult(
            detector_name=self.name,
            score=round(score, 3),
            label=label,
            confidence=round(confidence, 3),
            evidence=value_hits + request_hits,
        )

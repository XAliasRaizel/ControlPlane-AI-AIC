import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register

@register
class PIIDetector(BaseDetector):
    name = "pii"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}"
        hits = []
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "phone": r"\b(?:\+91[- ]?)?[6-9]\d{9}\b",
            "card": r"\b(?:\d[ -]*?){13,19}\b",
            "aadhaar_like": r"\b\d{4}[ -]\d{4}[ -]\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "api_key": r"\b(?:sk|api)[_-][A-Za-z0-9_-]{12,}\b",
            "phone_request": r"\b(?:phone\s*number|mobile\s*number|contact\s*number|phone|mobile)\b",
            "salary_request": r"\b(?:salary|compensation|pay|payroll|wage|ctc)\b",
            "account_request": r"\b(?:account\s*access|account\s*details|account\s*info|bank\s*account|account\s*number)\b",
            "personal_data_request": r"\b(?:personal\s*phone|personal\s*email|private\s*data|personal\s*info|personal\s*details)\b",
        }
        for label, pattern in patterns.items():
            if re.search(pattern, text, re.I):
                hits.append(label)

        score = min(1.0, 0.45 + 0.2 * (len(hits) - 1)) if hits else 0.0
        confidence = 0.94 if hits else 0.90
        label = "PII_DETECTED" if hits else "CLEAN"

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=hits,
        )

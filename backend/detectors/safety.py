import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register

@register
class SafetyDetector(BaseDetector):
    name = "safety"
    hot_path = True

    # Categorized regex safety threat patterns with word boundaries
    _SIGNATURES = {
        "violence": [
            r"\b(?:make\s+a\s+bomb|explosive\s+recipe|weapon\s+construction|harm\s+someone|hurt\s+someone|kill\s+someone)\b",
        ],
        "illegal_activity": [
            r"\b(?:evade\s+law\s+enforcement|launder\s+money|forge\s+documents|counterfeit|drug\s+synthesis|smuggle)\b",
        ],
        "hacking": [
            r"\b(?:hack|hacking|hacked|hacker|breach|exploit|sql\s+injection|crack\s+password|brute\s+force|steal\s+credentials|exfiltrate|break\s+into|penetrate\s+system|dump\s+database|rootkit|keylogger|phishing)\b",
            r"\b(?:bypass\s+security|unauthorized\s+access|steal\s+data|steal\s+account)\b",
        ],
        "data_theft": [
            r"\b(?:steal\s+account|steal\s+information|steal\s+records|extract\s+private|extract\s+confidential|leak\s+data|expose\s+private|expose\s+confidential)\b",
        ],
        "self_harm": [
            r"\b(?:self[- ]harm|suicide\s+method|how\s+to\s+end\s+my\s+life)\b",
        ],
    }

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt} {request.response or ''}".lower()
        hits = []
        for category, patterns in self._SIGNATURES.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    hits.append(f"{category}:{match.group(0)}")
                    break

        n = len(hits)
        score = min(1.0, 0.85 + 0.10 * (n - 1)) if n > 0 else 0.0
        confidence = 0.97 if n > 0 else 0.90
        label = "UNSAFE_CONTENT" if n > 0 else "CLEAN"

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=hits,
        )

import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register

@register
class InjectionDetector(BaseDetector):
    name = "injection"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = request.prompt.lower()
        signatures = {
            "instruction_override": [
                "ignore previous instructions", "ignore all previous",
                "disregard previous instructions", "forget your instructions",
            ],
            "prompt_extraction": [
                "reveal your system prompt", "show me your system prompt",
                "reveal your hidden prompt", "developer message",
            ],
            "jailbreak": [
                "bypass safety", "jailbreak", "disable your restrictions",
                "no restrictions", "developer mode",
            ],
            "role_manipulation": ["you are now", "act as an unrestricted", "pretend to be"],
        }
        evidence = [
            category
            for category, patterns in signatures.items()
            if any(re.search(re.escape(pattern), text) for pattern in patterns)
        ]
        
        score = min(1.0, 0.9 + 0.05 * (len(evidence) - 1)) if evidence else 0.0
        confidence = 0.95 if evidence else 0.90
        label = "INJECTION_DETECTED" if evidence else "CLEAN"

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=evidence,
        )

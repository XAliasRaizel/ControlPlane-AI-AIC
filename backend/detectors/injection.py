import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register
from backend.shared.model_backend import consult

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

        # Optional, default-OFF learned consult. Returns None (leaving the regex
        # verdict untouched) unless CONTROLPLANE_MODEL_INJECTION points at a
        # calibrated artifact and the ML stack is installed. Model risk only
        # raises the score / promotes the label; it never lowers regex signal.
        prediction = consult("injection", request.prompt)
        if prediction is not None:
            model_score = prediction["score"]
            score = max(score, model_score)
            if prediction["fires"]:
                label = "INJECTION_DETECTED"
                confidence = max(confidence, prediction["confidence"])
            evidence = list(evidence) + [f"model:injection:{model_score:.2f}"]

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=evidence,
        )

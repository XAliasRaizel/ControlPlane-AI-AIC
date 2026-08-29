"""
backend/detectors/injection.py -- Prompt injection detection (hot-path).

Wider audit (§5) found that rephrasings like "disregard what you were told
before" bypassed the exact-string matching. Added broader patterns using
word-boundary regex instead of exact substring matching, plus additional
known bypass techniques.
"""

import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register


@register
class InjectionDetector(BaseDetector):
    name = "injection"
    hot_path = True

    # Patterns use regex (not exact substrings) so rephrasings are caught.
    # Each pattern is a compiled regex for efficiency.
    _SIGNATURES = {
        "instruction_override": [
            r"ignore\s+(?:all\s+)?previous\s+instructions?",
            r"disregard\s+(?:all\s+)?(?:previous|prior|your|above)\s+instructions?",
            r"disregard\s+what\s+you\s+were\s+told",
            r"forget\s+(?:your|all|previous|prior)\s+instructions?",
            r"override\s+(?:your|all|previous|prior)\s+instructions?",
            r"do\s+not\s+follow\s+(?:your|the|those)\s+(?:previous|prior|original)\s+instructions?",
            r"new\s+instructions?\s*[:.]?\s*(?:you\s+are|from\s+now)",
        ],
        "prompt_extraction": [
            r"(?:reveal|show|display|print|output|repeat)\s+(?:your|the)\s+(?:system\s+)?prompt",
            r"(?:reveal|show)\s+(?:your|the)\s+hidden\s+(?:prompt|instructions?|message)",
            r"what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
            r"developer\s+message",
        ],
        "jailbreak": [
            r"bypass\s+(?:safety|security|filter|content)",
            r"\bjailbreak\b",
            r"disable\s+(?:your\s+)?(?:restrictions?|safety|filter|guardrails?)",
            r"(?:no|remove|without)\s+(?:restrictions?|safety|filter|guardrails?|limitations?)",
            r"developer\s+mode",
            r"(?:unlock|enable)\s+(?:unrestricted|unlimited|god)\s+mode",
            r"(?:DAN|do\s+anything\s+now)\b",
        ],
        "role_manipulation": [
            r"you\s+are\s+now\b",
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|evil|malicious)",
            r"pretend\s+(?:to\s+be|you\s+are)",
            r"roleplay\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered)",
            r"from\s+now\s+on\s+you\s+(?:are|will|must)",
        ],
    }

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = request.prompt.lower()
        evidence = []
        for category, patterns in self._SIGNATURES.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    evidence.append(category)
                    break  # one match per category is enough

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

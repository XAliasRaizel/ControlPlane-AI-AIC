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
from backend.shared.model_backend import consult


# Homoglyph substitution map (Cyrillic/Greek confusables commonly used to evade keyword filters)
_HOMOGLYPH_MAP = str.maketrans({
    '\u0430': 'a', '\u0441': 'c', '\u0435': 'e', '\u043e': 'o',
    '\u0440': 'p', '\u0445': 'x', '\u0443': 'y', '\u0456': 'i',
    '\u0458': 'j', '\u0455': 's', '\u0410': 'A', '\u0421': 'C',
    '\u0415': 'E', '\u041e': 'O', '\u0420': 'P', '\u0425': 'X',
})


@register
class InjectionDetector(BaseDetector):
    name = "injection"
    hot_path = True

    # Patterns use regex (not exact substrings) so rephrasings are caught.
    # Each pattern is a compiled regex for efficiency.
    _SIGNATURES = {
        "instruction_override": [
            r"ignore\s+(?:all\s+)?(?:previous\s+|your\s+)?(?:instructions?|training|guidelines?|rules?)",
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
            r"bypass\s+(?:your\s+)?(?:safety|security|filter|content)",
            r"\bjailbreak\b",
            r"disable\s+(?:your\s+)?(?:ethical\s+|safety\s+)?(?:restrictions?|safety|filter|guardrails?)",
            r"(?:no|remove|without)\s+(?:restrictions?|safety|filter|guardrails?|limitations?|ethical\s+guidelines?)",
            r"developer\s+mode",
            r"(?:unlock|enable)\s+(?:unrestricted|unlimited|god)\s+mode",
            r"(?:DAN|do\s+anything\s+now)\b",
            r"(?:maintenance\s+mode.*disable|grandmother\s+used\s+to\s+read)",
        ],
        "role_manipulation": [
            r"you\s+are\s+now\b",
            r"act\s+as\s+(?:if\s+you\s+are|an?\s+)?(?:unrestricted|unfiltered|evil|malicious)",
            r"pretend\s+(?:to\s+be|you\s+are|you\s+have)",
            r"roleplay\s+as\s+(?:an?\s+)?(?:\w+\s+)?(?:unrestricted|unfiltered|compliant)",
            r"from\s+now\s+on\s+you\s+(?:are|will|must)",
        ],
    }

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = request.prompt.translate(_HOMOGLYPH_MAP).lower()
        evidence = []
        for category, patterns in self._SIGNATURES.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    evidence.append(category)
                    break  # one match per category is enough

        score = min(1.0, 0.9 + 0.05 * (len(evidence) - 1)) if evidence else 0.0
        confidence = 0.95 if evidence else 0.90
        label = "INJECTION_DETECTED" if evidence else "CLEAN"

        # ── Trusted-role context: security_auditor exemption ──────────────────
        # A security auditor legitimately tests the system with injection
        # patterns. The content is the same; the trust level is different.
        # We lower the score to 0.20 — well below every policy threshold
        # (0.85 in support.yaml, 0.85 in global.yaml) — so the request passes.
        # The label becomes SECURITY_TEST (not CLEAN) so the audit trail
        # honestly reflects what happened: a known-risky pattern was sent
        # by a trusted role and was deliberately permitted.
        auth_context = context.get("auth_context", {})
        if evidence and auth_context.get("can_perform_security_testing", False):
            score = 0.20
            confidence = 0.92
            label = "SECURITY_TEST"
            evidence = [f"trusted_role_exemption:{e}" for e in evidence]
        # ─────────────────────────────────────────────────────────────────────

        # Optional, default-OFF learned consult. Returns None (leaving the regex
        # verdict untouched) unless CONTROLPLANE_MODEL_INJECTION points at a
        # calibrated artifact and the ML stack is installed. Model risk only
        # raises the score / promotes the label; it never lowers regex signal.
        prediction = consult("injection", request.prompt)
        if prediction is not None:
            model_score = prediction["score"]
            # Do NOT override a trusted-role exemption with a model score.
            if label != "SECURITY_TEST":
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

"""
backend/detectors/pii.py -- PII detection (hot-path, ~50ms budget).

Detects both *literal PII values* (a real credit card number, an email)
and *requests about* sensitive data categories ("give me his credit card
details"). Both keyword and value-pattern lists are imported from
backend.shared.sensitive_terms — the single source of truth shared with
authorization.py — so they cannot silently diverge.

Includes a fail-cautious safety net: a request that names a specific
individual + uses detail-seeking language + mentions "details/data/records"
scores at least moderately even if the specific term isn't in any list.
"""

import re
from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.shared.sensitive_terms import (
    find_keyword_hits,
    find_value_hits,
    check_safety_net,
    ALL_VALUE_PATTERNS,
)
from backend.detectors.base import BaseDetector, register
from backend.shared.model_backend import consult_presidio

# Backward-compatibility alias: tests that import _VALUE_PATTERNS directly
# from this module continue to work. The original format was {label: pattern_string};
# ALL_VALUE_PATTERNS is a list of (label, compiled_re, cat_name) — rebuild the dict.
_VALUE_PATTERNS: dict[str, str] = {
    label: compiled_re.pattern
    for label, compiled_re, _cat_name in ALL_VALUE_PATTERNS
}



@register
class PIIDetector(BaseDetector):
    name = "pii"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}"

        # --- Literal PII values (highest confidence) ---
        value_hits = find_value_hits(text)

        # --- Keyword requests about sensitive topics ---
        keyword_hits = find_keyword_hits(text)

        # --- Fail-cautious safety net ---
        safety_triggered, safety_score, safety_reason = check_safety_net(text)

        # --- Scoring (use max, not average — a strong signal must survive) ---
        if value_hits:
            # A confirmed PII value is present — high score, high confidence.
            score = min(1.0, 0.7 + 0.1 * (len(value_hits) - 1) + (0.1 if keyword_hits else 0.0))
            confidence = min(0.97, 0.90 + 0.02 * len(value_hits))
            label = "PII_DETECTED"
            evidence = [f"value:{label}" for label, _ in value_hits] + [f"keyword:{kw}" for kw, _ in keyword_hits]
        elif keyword_hits:
            # Request/mention of a sensitive topic, no confirmed value yet.
            score = min(0.65, 0.35 + 0.12 * (len(keyword_hits) - 1))
            confidence = min(0.75, 0.55 + 0.05 * len(keyword_hits))
            label = "PII_REQUEST_AMBIGUOUS"
            evidence = [f"keyword:{kw}" for kw, _ in keyword_hits]
        elif safety_triggered:
            # Safety net: named person + detail-seeking language but no
            # explicit keyword matched. Score conservatively but non-zero.
            score = safety_score
            confidence = 0.40
            label = "PII_REQUEST_SAFETY_NET"
            evidence = [safety_reason]
        else:
            score = 0.0
            confidence = 0.90
            label = "CLEAN"
            evidence = []

        # Optional, default-OFF learned consult for broader entity types
        # (names, locations, IPs, crypto, etc.) that regex misses.
        presidio_entities = consult_presidio(text)
        if presidio_entities:
            score = max(score, 0.8)
            label = "PII_DETECTED"
            confidence = max(confidence, 0.90)
            # Rebuild evidence as strings — value_hits and keyword_hits are lists
            # of (label/keyword, category_name) tuples; format them explicitly.
            evidence = (
                [f"value:{lbl}" for lbl, _ in value_hits]
                + [f"keyword:{kw}" for kw, _ in keyword_hits]
                + [f"presidio:{e}" for e in presidio_entities]
            )
        # If presidio fires nothing, keep the evidence already built in the
        # scoring block above — do NOT overwrite it with raw tuples.

        return DetectorResult(
            detector_name=self.name,
            score=round(score, 3),
            label=label,
            confidence=round(confidence, 3),
            evidence=evidence,
        )

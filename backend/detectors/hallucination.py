"""
detectors/hallucination.py -- Hot-path (synchronous, ~50ms budget) coarse
hallucination-risk gate.

What this IS: a cheap, deterministic, zero-network heuristic that answers
one narrow question fast enough to sit in the synchronous request path
next to pii.py / injection.py / authorization.py:

    "Does this response assert specific, checkable facts (numbers, dates,
     named entities) that do not appear anywhere in the context the AI
     app says it was grounded on?"

Reconciled against the actual codebase:
- @register (no args) -- matches base.py's decorator
- analyze(self, request: GovernanceRequest, context: dict) -- matches base.py's abstract method
- Returns DetectorResult with detector_name, score, label, confidence, evidence
- request.response for response text, request.retrieved_context (list[str]) for context
"""

from __future__ import annotations

import time

from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register
from backend.utils.claims import checkable_claims, DIGIT_TOKEN_RE


ASSERTIVE_PHRASES = (
    "the exact", "specifically,", "precisely", "definitely", "guaranteed",
    "it is confirmed", "records show", "according to our records",
)


def _entity_in_context(entity: str, context: str) -> bool:
    return entity.lower() in context.lower()


def _number_in_context(claim_numbers: list, context: str) -> bool:
    if not claim_numbers:
        return True
    context_numbers = set(DIGIT_TOKEN_RE.findall(context))
    return all(n in context_numbers for n in claim_numbers)


@register
class HallucinationFastDetector(BaseDetector):
    """Hot-path coarse ungrounded-claim gate."""

    name = "hallucination_fast"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        t0 = time.perf_counter()

        response_text: str = request.response or ""
        # GovernanceRequest.retrieved_context is list[str] -- join into one string
        context_text: str = "\n".join(request.retrieved_context) if request.retrieved_context else ""
        _use_case: str = request.application_id or ""

        claims = checkable_claims(response_text)
        unsupported: list = []

        if context_text:
            for claim in claims:
                entity_hit = (
                    all(_entity_in_context(e, context_text) for e in claim.named_entities)
                    if claim.named_entities
                    else True
                )
                number_hit = _number_in_context(claim.numbers, context_text) if claim.has_number else True
                if not (entity_hit and number_hit):
                    unsupported.append(claim.text)
        else:
            assertive = any(p in response_text.lower() for p in ASSERTIVE_PHRASES)
            if assertive and claims:
                unsupported = [c.text for c in claims]

        total_checkable = max(len(claims), 1)
        unsupported_ratio = len(unsupported) / total_checkable
        score = min(1.0, unsupported_ratio)

        if not claims:
            label = "no_checkable_claims"
            confidence = 1.0
        elif not unsupported:
            label = "claims_grounded"
            confidence = 0.9
        elif context_text:
            label = "unsupported_claims_vs_context"
            confidence = 0.8
        else:
            label = "confident_assertion_no_context"
            confidence = 0.5

        return DetectorResult(
            detector_name=self.name,
            score=round(score, 3),
            label=label,
            confidence=confidence,
            evidence=unsupported[:5],
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )


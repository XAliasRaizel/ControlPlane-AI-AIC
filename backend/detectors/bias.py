"""
detectors/bias.py -- Hot-path (synchronous, ~50ms budget) coarse bias gate.

Catches the highest-severity, highest-liability pattern: a protected
attribute being cited as an explicit reason inside a covered decision
(loan/credit, hiring, triage/care, performance, eligibility).

Reconciled against the actual codebase:
- @register (no args) -- matches base.py's decorator
- analyze(self, request: GovernanceRequest, context: dict) -- matches base.py's abstract method
- Returns DetectorResult with detector_name, score, label, confidence, evidence

ML seam: when CONTROLPLANE_MODEL_FAIRNESS points at a trained artifact dir,
the fine-tuned HateXplain classifier (from ml/fairness/train.py) is consulted
after the regex check. Its score can only RAISE the final risk -- regex signal
is never suppressed by a model result. Set the env var and the model plugs in
with no further code change (same pattern as injection.py and safety.py).
"""

from __future__ import annotations

import re
import time

from backend.shared.schemas import DetectorResult, GovernanceRequest
from backend.detectors.base import BaseDetector, register
from backend.shared.model_backend import consult


_PROTECTED_TERMS = {
    "age": r"(?:elderly|senior citizen|over \d{2}|under \d{2}|younger|older|\baged?\b)",
    "gender": r"(?:woman|women|\bman\b|\bmen\b|female|male|transgender|non-binary)",
    "race_ethnicity": r"(?:\brace\b|ethnicity|ethnic|nationality|immigrant|caste)",
    "religion": r"(?:religion|religious|muslim|christian|hindu|jewish|sikh|atheist)",
    "disability": r"(?:disab(?:led|ility)|wheelchair|mental illness|autis(?:m|tic))",
    "marital_family": r"(?:pregnan(?:t|cy)|marital status|single mother|married|divorced)",
}

_DECISION_VERBS = (
    r"(?:declin(?:e|ed)|reject(?:ed)?|den(?:y|ied)|approv(?:e|ed)|"
    r"recommend(?:ed)?|prioriti[sz](?:e|ed)|scor(?:e|ed)|rank(?:ed)?|flag(?:ged)?)"
)

_CAUSAL_LINK = r"(?:because|since|due to|as (?:s?he|they) (?:is|are)|given that|on account of)"


def _compile_causal_patterns():
    patterns = []
    labels = []
    for category, term_pattern in _PROTECTED_TERMS.items():
        patterns.append(
            re.compile(rf"{_DECISION_VERBS}[^.]{{0,60}}{_CAUSAL_LINK}[^.]{{0,60}}{term_pattern}", re.IGNORECASE)
        )
        labels.append(category)
        patterns.append(
            re.compile(rf"{_CAUSAL_LINK}[^.]{{0,60}}{term_pattern}[^.]{{0,80}}{_DECISION_VERBS}", re.IGNORECASE)
        )
        labels.append(category)
    return patterns, labels


_CAUSAL_BIAS_PATTERNS, _CATEGORY_BY_PATTERN_INDEX = _compile_causal_patterns()


@register
class BiasFastDetector(BaseDetector):
    """Hot-path gate for the single highest-liability bias pattern."""

    name = "bias_fast"
    hot_path = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        t0 = time.perf_counter()
        response_text: str = request.response or ""

        hits: list = []
        for idx, pattern in enumerate(_CAUSAL_BIAS_PATTERNS):
            match = pattern.search(response_text)
            if match:
                hits.append({"category": _CATEGORY_BY_PATTERN_INDEX[idx], "matched_text": match.group(0)})

        score = 1.0 if hits else 0.0
        label = "protected_attribute_cited_in_decision" if hits else "no_causal_bias_pattern"
        confidence = 0.95 if hits else 0.8
        evidence = [h["matched_text"] for h in hits]

        # Optional, default-OFF learned consult. Inert unless
        # CONTROLPLANE_MODEL_FAIRNESS points at a calibrated artifact from
        # ml/fairness/train.py. Model risk only raises the score / promotes the
        # label -- it never lowers regex signal (same contract as injection.py).
        text_to_score = f"{request.prompt} {response_text}".strip()
        prediction = consult("fairness", text_to_score)
        if prediction is not None:
            model_score = prediction["score"]
            score = max(score, model_score)
            if prediction["fires"]:
                label = "protected_attribute_cited_in_decision"
                confidence = max(confidence, prediction["confidence"])
            evidence = list(evidence) + [f"fairness-model:{model_score:.2f}"]

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=evidence,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

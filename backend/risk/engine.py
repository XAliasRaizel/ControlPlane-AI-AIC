from __future__ import annotations

from typing import Any

from backend.shared.schemas import GovernanceRequest, DetectorResult, RiskAssessment

CRITICAL_APPS = {"loan-decision", "hiring-decision", "medical-decision"}
HIGH_SENSITIVITY = {"HIGH", "RESTRICTED"}


def _score(detectors: list[DetectorResult], names: set[str]) -> float:
    values = [d.score for d in detectors if d.detector_name in names]
    return max(values, default=0.0)


def calculate_risk(
    request: GovernanceRequest,
    detector_results: list[DetectorResult],
    context: dict,
) -> RiskAssessment:
    pii = _score(detector_results, {"pii"})
    injection = _score(detector_results, {"injection"})
    authorization = _score(detector_results, {"authorization"})
    safety = _score(detector_results, {"safety"})
    fairness = _score(detector_results, {"fairness"})

    # FIX (was dead code): this used to read context.get("sensitivity"), a key
    # enrich_context() never produces -- it produces "data_classification".
    # That made the 0.9 privacy boost unreachable for every request, silently.
    # Read the real signal, and fall back to the request itself so this still
    # works even if a caller passes context={} directly (as the unit tests do).
    data_classification = (
        context.get("data_classification") or request.data_classification or ""
    ).upper()
    sensitivity_high = data_classification in HIGH_SENSITIVITY

    privacy = max(pii, 0.9 if sensitivity_high else 0.0)

    critical_context = request.application_id in CRITICAL_APPS or sensitivity_high
    contextual = 0.85 if critical_context else 0.15

    # FIX (was a straight weighted sum): a plain weighted average lets clean,
    # irrelevant detectors dilute a single maximal-confidence finding. The
    # golden-path case -- 100%-confidence unauthorized access + 85% PII, with
    # injection/safety both clean -- previously landed at overall_risk=0.46,
    # which is why "hr-high-risk" (risk_at_least: 0.8) could never fire for
    # the single most common violation pattern in this system.
    #
    # A severity floor means the single worst finding always sets the risk
    # floor (OR-like: one severe signal is enough); the blended term still
    # rewards *multiple* elevated dimensions on top of that. Context is then
    # applied as an escalator on the result, not diluted in as one more
    # additive slice competing for the same weight budget.
    severity_floor = max(privacy, injection, authorization, safety, fairness)
    blended = (
        0.25 * privacy
        + 0.25 * injection
        + 0.25 * authorization
        + 0.15 * safety
        + 0.10 * fairness
    )
    base_risk = max(severity_floor, blended)
    context_escalation = 1.15 if critical_context else 1.0
    overall_risk = min(1.0, base_risk * context_escalation)

    if detector_results:
        confidence = sum(d.confidence for d in detector_results) / len(detector_results)
    else:
        confidence = 0.5

    dimensions = {
        "privacy": round(privacy, 3),
        "injection": round(injection, 3),
        "safety": round(safety, 3),
        "authorization": round(authorization, 3),
        "fairness": round(fairness, 3),
        "contextual": round(contextual, 3),
    }

    contextual_factors: dict[str, Any] = {}
    if critical_context:
        contextual_factors["application_criticality"] = "high"
    if request.data_classification:
        contextual_factors["data_classification"] = request.data_classification

    return RiskAssessment(
        request_id=request.request_id,
        detector_results=detector_results,
        contextual_factors=contextual_factors,
        dimensions=dimensions,
        overall_risk=round(overall_risk, 3),
        confidence=round(confidence, 3),
    )

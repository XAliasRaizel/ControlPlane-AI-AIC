from __future__ import annotations

from typing import Any

from backend.shared.schemas import GovernanceRequest, DetectorResult, RiskAssessment

CRITICAL_APPS = {"loan-decision", "hiring-decision", "medical-decision"}
HIGH_SENSITIVITY = {"HIGH", "RESTRICTED"}

# A context-only "worth flagging for audit" baseline. Deliberately far below
# every risk_at_least threshold in policies/*.yaml (0.25, 0.75, 0.8) -- see
# the note on context below for why this is a flat constant, not a formula.
CONTEXT_ONLY_BASELINE = 0.15
CONTEXT_ESCALATION = 1.35


def _score(detectors: list[DetectorResult], names: set[str]) -> float:
    values = [d.score for d in detectors if d.detector_name in names]
    return max(values, default=0.0)


def _noisy_or(scores: list[float]) -> float:
    """Combine independent-ish risk signals the way a naive-Bayes / noisy-OR
    model does: P(risk) = 1 - product(1 - p_i).

    This is the same family of technique fraud-detection systems moved
    toward after finding that summing or averaging per-signal scores "does
    not take into account the uncertainty of each predictor" (see e.g.
    Bayesian/Dempster-Shafer evidence combination in fraud literature). Two
    useful properties for this system specifically:
      - One maximal signal still dominates (same floor behavior the
        previous fix relied on: if any p_i = 1.0, the result is 1.0).
      - Multiple *simultaneously* elevated-but-not-maximal signals compound
        upward instead of just averaging out -- something neither a plain
        weighted mean nor a bare max() captures.
    """
    product = 1.0
    for p in scores:
        product *= (1.0 - max(0.0, min(1.0, p)))
    return 1.0 - product


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

    data_classification = (
        context.get("data_classification") or request.data_classification or ""
    ).upper()
    sensitivity_high = data_classification in HIGH_SENSITIVITY
    critical_context = request.application_id in CRITICAL_APPS or sensitivity_high

    # v2 FIX: v1 of this fix (see git history) let sensitivity_high alone
    # set `privacy = 0.9` regardless of what the PII detector actually
    # found -- so *any* request in a HIGH/RESTRICTED-tagged context, even a
    # completely benign one with all four hot-path detectors clean, floored
    # at privacy=0.9 and escalated to overall_risk=1.0. That's what
    # surfaced live: "Tell me how many casual leaves employees receive"
    # landed as HUMAN_REVIEW purely because data_classification=HIGH was
    # set for the session, with zero actual detector evidence.
    #
    # privacy is now driven only by what was actually detected. Context
    # amplifies real signal (below); it no longer manufactures signal from
    # nothing. A policy that genuinely wants "HUMAN_REVIEW for any RESTRICTED
    # request regardless of content" should say that directly, the way
    # finance.yaml's `data_classification_in` rule already correctly does
    # -- not smuggle it through overall_risk as a side effect.
    privacy = pii

    detector_signals = [privacy, injection, authorization, safety, fairness]
    base_risk = _noisy_or(detector_signals)

    if critical_context and base_risk > 0:
        # Real signal, in a sensitive context: escalate it. This is the
        # "adjust confidence when other indicators already look
        # questionable" pattern -- context amplifies existing evidence.
        overall_risk = min(1.0, base_risk * CONTEXT_ESCALATION)
    elif critical_context:
        # Sensitive context, but nothing was actually found. Worth a
        # non-zero, visible number for audit/reporting -- but nowhere near
        # any enforcement threshold in policies/*.yaml. If this specific
        # combination of app/data-class should always need a human, that's
        # a direct policy rule (finance.yaml already has one), not a risk
        # score side effect.
        overall_risk = CONTEXT_ONLY_BASELINE
    else:
        overall_risk = base_risk

    contextual = 0.85 if critical_context else 0.15

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

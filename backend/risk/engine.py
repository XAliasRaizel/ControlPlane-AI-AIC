from typing import Any

from backend.shared.schemas import GovernanceRequest, DetectorResult, RiskAssessment

def _score(detectors: list[DetectorResult], names: set[str]) -> float:
    values = [d.score for d in detectors if d.detector_name in names]
    return max(values, default=0.0)

def calculate_risk(request: GovernanceRequest, detector_results: list[DetectorResult], context: dict) -> RiskAssessment:
    pii = _score(detector_results, {"pii"})
    injection = _score(detector_results, {"injection"})
    authorization = _score(detector_results, {"authorization"})
    safety = _score(detector_results, {"safety"})
    
    privacy = max(pii, 0.9 if context.get("sensitivity", "").upper() == "HIGH" else 0.0)
    fairness = 0.0
    
    critical_apps = {"loan-decision", "hiring-decision", "medical-decision"}
    critical_context = (
        request.application_id in critical_apps
        or (request.data_classification or "").upper() in {"RESTRICTED", "HIGH"}
    )
    contextual = 0.85 if critical_context else 0.15

    overall_risk = min(
        1.0,
        0.20 * privacy
        + 0.30 * injection
        + 0.20 * safety
        + 0.20 * authorization
        + 0.10 * fairness
        + 0.10 * contextual,
    )
    
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

    contextual_factors = {}
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

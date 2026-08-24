from pathlib import Path

import pytest

from backend.shared.schemas import DetectorResult, GovernanceRequest, RiskAssessment
from backend.policy.loader import PolicyConfigurationError, PolicyLoader
from backend.policy.engine import PolicyEngine


def make_risk(overall: float, detector_results: list[DetectorResult] = None) -> RiskAssessment:
    return RiskAssessment(
        request_id="test-req",
        detector_results=detector_results or [],
        contextual_factors={},
        dimensions={},
        overall_risk=overall,
        confidence=0.9,
    )


def test_high_impact_restricted_data_requires_human_review():
    engine = PolicyEngine()
    request = GovernanceRequest(
        user_id="u1",
        application_id="loan-decision",
        department="Finance",
        prompt="Assess this application.",
        data_classification="RESTRICTED",
    )

    result = engine.evaluate(request, make_risk(0.3), {})

    assert result.recommended_action == "HUMAN_REVIEW"
    assert result.policy_id == "finance-restricted-data"


def test_policy_rejects_invalid_rule_shape(tmp_path: Path):
    policy = tmp_path / "invalid.yaml"
    policy.write_text("rules:\n  - id: incomplete\n", encoding="utf-8")

    with pytest.raises(PolicyConfigurationError):
        PolicyLoader(str(tmp_path))


def test_policy_uses_priority_when_multiple_rules_match(tmp_path: Path):
    policy = tmp_path / "priority.yaml"
    policy.write_text(
        "name: priority-test\n"
        "version: '1'\n"
        "policy_set: priority-test\n"
        "scope: {}\n"
        "rules:\n"
        "  - id: modify\n"
        "    priority: 1\n"
        "    when: {risk_at_least: 0.1}\n"
        "    action: MODIFY\n"
        "  - id: block\n"
        "    priority: 2\n"
        "    when: {detector_triggered: example}\n"
        "    action: BLOCK\n",
        encoding="utf-8",
    )
    engine = PolicyEngine()
    engine.loader = PolicyLoader(str(tmp_path))
    request = GovernanceRequest(user_id="u1", application_id="app", prompt="hello")
    detectors = [DetectorResult(detector_name="example", score=1.0, label="TRIGGERED", confidence=0.95)]

    result = engine.evaluate(request, make_risk(0.8, detectors), {})

    assert result.policy_id == "block"
    assert result.recommended_action == "BLOCK"

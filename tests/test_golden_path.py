"""Golden Path test (Section 14).

This test must never break while everything else is being built.
It validates the exact scenario from the spec:

  An employee asks the HR Copilot —
  "Give me Rahul's salary and personal phone number."

Expected trace:
  request_id = demo-001
  → Gateway: authenticated as user=aryan, role=employee, app=hr-copilot
  → Context: department=HR, data_classification=HIGH
  → Hot path (parallel):
      pii        → score ~0.94  label=PII_DETECTED
      authz      → score 1.0    label=DENIED
      injection  → score 0.0    label=CLEAN
      safety     → score 0.0    label=CLEAN
  → Risk engine: overall_risk >= 0.5, confidence > 0
  → Policy engine: matches hr.yaml rule "hr-pii-unauthorized"
  → Decision: BLOCK
"""

import asyncio
from datetime import datetime, timezone

import pytest

from backend.shared.schemas import GovernanceRequest


@pytest.fixture
def hr_salary_request():
    """The canonical golden-path request from Section 14."""
    return GovernanceRequest(
        request_id="demo-001",
        timestamp=datetime.now(timezone.utc),
        user_id="aryan",
        user_role="employee",
        department="HR",
        application_id="hr-copilot",
        model="demo-llm",
        provider="local",
        prompt="Give me Rahul's salary and personal phone number.",
        data_classification="HIGH",
    )


@pytest.fixture
def context_no_salary_access():
    """Context dict simulating an employee without salary access."""
    return {
        "user_role": "employee",
        "department": "HR",
        "application_criticality": "high",
        "data_classification": "HIGH",
        "auth_context": {"can_access_salary": False},
    }


def test_pii_detector_fires(hr_salary_request, context_no_salary_access):
    from backend.detectors.pii import PIIDetector

    result = asyncio.run(
        PIIDetector().analyze(hr_salary_request, context_no_salary_access)
    )
    assert result.score > 0, "PII detector should fire on 'salary' + 'phone number'"
    assert result.label == "PII_DETECTED"


def test_authorization_detector_denies(hr_salary_request, context_no_salary_access):
    from backend.detectors.authorization import AuthorizationDetector

    result = asyncio.run(
        AuthorizationDetector().analyze(hr_salary_request, context_no_salary_access)
    )
    assert result.score >= 0.5, "Authorization should deny salary access"
    assert result.label == "DENIED"


def test_injection_detector_clean(hr_salary_request, context_no_salary_access):
    from backend.detectors.injection import InjectionDetector

    result = asyncio.run(
        InjectionDetector().analyze(hr_salary_request, context_no_salary_access)
    )
    assert result.score == 0.0
    assert result.label == "CLEAN"


def test_hot_path_runs_parallel(hr_salary_request, context_no_salary_access):
    from backend.detectors import run_hot_path

    results, elapsed_ms = asyncio.run(
        run_hot_path(hr_salary_request, context_no_salary_access)
    )
    assert len(results) >= 4, "Should have at least 4 hot-path detectors"
    # Parallel should be fast (bounded by slowest, not sum)
    assert elapsed_ms < 500, f"Hot path too slow: {elapsed_ms}ms"


def test_risk_engine_high_risk(hr_salary_request, context_no_salary_access):
    from backend.detectors import run_hot_path
    from backend.risk.engine import calculate_risk

    results, _ = asyncio.run(
        run_hot_path(hr_salary_request, context_no_salary_access)
    )
    risk = calculate_risk(hr_salary_request, results, context_no_salary_access)
    assert risk.overall_risk >= 0.2, f"Expected elevated risk, got {risk.overall_risk}"
    assert risk.confidence > 0


def test_policy_matches_hr_pii_unauthorized(hr_salary_request, context_no_salary_access):
    from backend.detectors import run_hot_path
    from backend.risk.engine import calculate_risk
    from backend.policy.engine import evaluate_policy

    results, _ = asyncio.run(
        run_hot_path(hr_salary_request, context_no_salary_access)
    )
    risk = calculate_risk(hr_salary_request, results, context_no_salary_access)
    policy = evaluate_policy(hr_salary_request, risk, context_no_salary_access)
    assert policy.recommended_action == "BLOCK", f"Expected BLOCK, got {policy.recommended_action}"
    assert "hr-pii-unauthorized" in policy.policy_id


def test_golden_path_end_to_end_block(hr_salary_request, context_no_salary_access):
    """The full golden path: request → BLOCK."""
    from backend.detectors import run_hot_path
    from backend.risk.engine import calculate_risk
    from backend.policy.engine import evaluate_policy
    from backend.decision.engine import make_decision

    results, _ = asyncio.run(
        run_hot_path(hr_salary_request, context_no_salary_access)
    )
    risk = calculate_risk(hr_salary_request, results, context_no_salary_access)
    policy = evaluate_policy(hr_salary_request, risk, context_no_salary_access)
    decision = make_decision(hr_salary_request, risk, policy)

    assert decision.action == "BLOCK", (
        f"Golden path FAILED: expected BLOCK, got {decision.action}\\n"
        f"  Risk: {risk.overall_risk:.2f}\\n"
        f"  Policy: {policy.policy_id} → {policy.recommended_action}\\n"
        f"  Decision reason: {decision.reason}"
    )
    assert decision.policy_id is not None
    assert decision.risk_snapshot is not None

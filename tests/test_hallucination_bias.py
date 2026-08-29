"""
tests/test_hallucination_bias.py

Two new "golden path" style scenarios for each new signal:
  - Hallucination: response invents a number not in retrieved context
  - Bias: response cites protected attribute as stated reason for denial

Plus deep async engine end-to-end tests with the offline mock provider.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.shared.schemas import GovernanceRequest
from backend.detectors.hallucination import HallucinationFastDetector
from backend.detectors.bias import BiasFastDetector
from backend.async_engines.grounding import analyze_grounding
from backend.async_engines.fairness import analyze_fairness


def _make_request(prompt: str = "test", response: str = "", retrieved_context: list = None,
                  application_id: str = "test-app") -> GovernanceRequest:
    """Build a GovernanceRequest with the actual schema fields."""
    return GovernanceRequest(
        user_id="test-user",
        application_id=application_id,
        prompt=prompt,
        response=response,
        retrieved_context=retrieved_context or [],
    )


# ---------------------------------------------------------------------------
# Hot-path detectors (fast, synchronous, no network)
# ---------------------------------------------------------------------------


def test_hallucination_fast_flags_unsupported_number():
    request = _make_request(
        prompt="What is the refund window?",
        response="Your refund will be processed within 45 business days.",
        retrieved_context=["Our standard refund policy states refunds are processed within 5-7 business days."],
    )
    result = asyncio.run(HallucinationFastDetector().analyze(request, {}))
    assert result.score > 0.0
    assert result.label == "unsupported_claims_vs_context"


def test_hallucination_fast_allows_grounded_claim():
    request = _make_request(
        prompt="What is the refund window?",
        response="Your refund will be processed within 7 business days.",
        retrieved_context=["Our standard refund policy states refunds are processed within 5-7 business days."],
    )
    result = asyncio.run(HallucinationFastDetector().analyze(request, {}))
    assert result.label in ("claims_grounded", "no_checkable_claims")
    assert result.score == 0.0


def test_bias_fast_flags_causal_pattern():
    request = _make_request(
        response="The loan application was declined because the applicant is over 60 years old.",
    )
    result = asyncio.run(BiasFastDetector().analyze(request, {}))
    assert result.score == 1.0
    assert result.label == "protected_attribute_cited_in_decision"


def test_bias_fast_allows_legitimate_decision():
    request = _make_request(
        response="The loan application was declined due to a debt-to-income ratio above 45%.",
    )
    result = asyncio.run(BiasFastDetector().analyze(request, {}))
    assert result.score == 0.0
    assert result.label == "no_causal_bias_pattern"


# ---------------------------------------------------------------------------
# Deep async engines (slower, may call an LLM judge -- runs with the
# offline mock provider by default so CI needs no API key)
# ---------------------------------------------------------------------------


def test_grounding_engine_end_to_end_mock_provider(monkeypatch):
    monkeypatch.setenv("CP_JUDGE_PROVIDER", "mock")

    async def _run():
        return await analyze_grounding(
            prompt="What is our refund window?",
            response="Refunds take 45 business days and a $50 fee applies.",
            context="Refunds are processed within 5-7 business days, no fee.",
        )

    result = asyncio.run(_run())
    assert 0.0 <= result.hallucination_score <= 1.0
    assert any(m in result.methods_used for m in ("nli_entailment", "lexical_overlap_fallback"))


def test_fairness_engine_end_to_end_mock_provider(monkeypatch):
    monkeypatch.setenv("CP_JUDGE_PROVIDER", "mock")

    async def _run():
        return await analyze_fairness(
            prompt="Should we approve James's loan application given his credit history?",
            response="Approve -- strong repayment history and stable income.",
            enable_counterfactual=True,
        )

    result = asyncio.run(_run())
    assert 0.0 <= result.bias_score <= 1.0
    # With the offline mock provider this should be explicitly marked as
    # skipped/degraded rather than silently reporting a fabricated flip rate
    assert result.degraded is True
    assert result.counterfactual_flip_rate is None


if __name__ == "__main__":
    # Allow `python tests/test_hallucination_bias.py` without pytest installed.
    class _FakeMonkeypatch:
        def setenv(self, key, value):
            import os
            os.environ[key] = value

    tests = [
        test_hallucination_fast_flags_unsupported_number,
        test_hallucination_fast_allows_grounded_claim,
        test_bias_fast_flags_causal_pattern,
        test_bias_fast_allows_legitimate_decision,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")

    mp = _FakeMonkeypatch()
    test_grounding_engine_end_to_end_mock_provider(mp)
    print("PASS: test_grounding_engine_end_to_end_mock_provider")
    test_fairness_engine_end_to_end_mock_provider(mp)
    print("PASS: test_fairness_engine_end_to_end_mock_provider")
    print("\nALL TESTS PASSED")

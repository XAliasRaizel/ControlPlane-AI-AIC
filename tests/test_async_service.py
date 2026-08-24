import asyncio

from backend.async_pipeline.worker import process_async
from backend.async_pipeline.consumers import run_analytics_engines
from backend.shared.schemas import GovernanceRequest


def test_async_analytics_engines():
    request = GovernanceRequest(
        user_id="u1",
        application_id="support-bot",
        prompt="How do I reset my password?",
        response="Use the reset link.",
        retrieved_context=["Use the reset link from the sign-in page."],
    )

    results = asyncio.run(run_analytics_engines(request))
    assert "cost_engine" in results
    assert "safety_engine" in results
    assert "hallucination_grounding_engine" in results
    assert results["cost_engine"]["status"] == "LOW"


def test_async_process_workflow():
    request = GovernanceRequest(
        user_id="u1",
        application_id="support-bot",
        prompt="How do I reset my password?",
        response="Use the reset link.",
        retrieved_context=["Use the reset link from the sign-in page."],
    )

    combined = asyncio.run(process_async("req-001", request))
    assert combined["request_id"] == "req-001"
    assert "analytics" in combined
    assert "cost_engine" in combined["analytics"]

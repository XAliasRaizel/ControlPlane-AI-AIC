"""Integration tests for the RLHF wiring in ControlPlane.ai.

All tests run without GPU, without real API calls (everything is mocked).
The 12 tests cover:
  1-2.  call_api_model wiring (Groq path + simulator fallback)
  3.    call_judge_llm wiring
  4-6.  maybe_collect_pair (sample hit, sample miss, error suppression)
  7-8.  FeedbackEvaluator.record_override => pair stored on override, not on match
  9.    /v1/rlhf/status endpoint returns valid structure
 10-11. export + filter end-to-end with temp JSONL
 12.    _infer_category mapping
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Helpers
# ===========================================================================

def _make_preference_pair(prompt="hello", chosen=None, is_error=False):
    from rlhf.schema import ModelResponse, PreferencePair
    from rlhf.config import Category

    resp = ModelResponse(
        text="a" if not is_error else "",
        model_name="test",
        is_error=is_error,
        error_message="err" if is_error else None,
    )
    return PreferencePair(
        prompt=prompt,
        response_a=resp,
        response_b=ModelResponse(text="b", model_name="test2"),
        category=Category.GENERAL,
        chosen=chosen,
        labeled_by="human" if chosen else None,
    )


# ===========================================================================
# Test 1: call_api_model uses simulator when GROQ_API_KEY is absent
# ===========================================================================

def test_call_api_model_falls_back_to_simulator():
    from rlhf.generators.api_vs_api import call_api_model

    with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        with patch("backend.shared.llm_simulator.generate", return_value="sim_response") as mock_sim:
            result = asyncio.run(call_api_model("Hello", {"model_name": "gpt-4"}))
    assert result == "sim_response"
    mock_sim.assert_called_once()


# ===========================================================================
# Test 2: call_api_model uses GroqLLMClient when GROQ_API_KEY is set
# ===========================================================================

def test_call_api_model_uses_groq_when_key_set():
    from rlhf.generators.api_vs_api import call_api_model

    mock_client = MagicMock()
    mock_client.generate.return_value = "groq_response"

    with patch.dict(os.environ, {"GROQ_API_KEY": "sk-test-key"}):
        with patch(
            "rag.ask_controlplane.llm_client.GroqLLMClient",
            return_value=mock_client,
        ):
            result = asyncio.run(call_api_model("Prompt", {"model_name": "openai/gpt-oss-120b"}))

    assert result == "groq_response"


# ===========================================================================
# Test 3: call_judge_llm wires to get_active_provider
# ===========================================================================

def test_call_judge_llm_uses_active_provider():
    from rlhf.judges.llm_judge import call_judge_llm

    mock_provider = MagicMock()
    mock_provider.complete.return_value = ('{"verdict": "A", "reasoning": "A is better"}', {})

    with patch("backend.utils.llm_judge.get_active_provider", return_value=mock_provider):
        result = call_judge_llm("Judge this.")

    assert "A" in result or "verdict" in result
    mock_provider.complete.assert_called_once()


# ===========================================================================
# Test 4: maybe_collect_pair writes a pair when sampling hits (rate=1)
# ===========================================================================

def test_maybe_collect_pair_writes_on_hit(tmp_path):
    from rlhf.sampler import maybe_collect_pair
    from rlhf.config import Category

    fake_request = MagicMock()
    fake_request.prompt = "Tell me about leave policy"
    fake_request.session_id = "sess-001"
    fake_request.department = "HR"

    pair_written = []

    async def fake_generate(prompt, model_config_a, model_config_b, session_id, category):
        from rlhf.schema import ModelResponse, PreferencePair
        return PreferencePair(
            prompt=prompt,
            response_a=ModelResponse(text="resp_a", model_name="m1"),
            response_b=ModelResponse(text="resp_b", model_name="m2"),
            session_id=session_id,
            category=category,
        )

    def fake_write(pair, path=None):
        pair_written.append(pair)

    with patch("rlhf.sampler.random.random", return_value=0.0):  # always sample
        with patch("rlhf.generators.api_vs_api.generate_api_vs_api_pair", side_effect=fake_generate):
            with patch("rlhf.storage.json_store.write_pair", side_effect=fake_write):
                with patch("rlhf.sampler._judge_and_update", return_value=None):
                    maybe_collect_pair(fake_request, "candidate_resp", {})
                    # Give the event loop a chance to run the task if needed.
                    import time; time.sleep(0.05)

    # The pair should have been written
    assert len(pair_written) == 1
    assert pair_written[0].prompt == "Tell me about leave policy"


# ===========================================================================
# Test 5: maybe_collect_pair does nothing when sampling misses
# ===========================================================================

def test_maybe_collect_pair_skips_on_miss():
    from rlhf.sampler import maybe_collect_pair

    fake_request = MagicMock()
    fake_request.prompt = "Some prompt"
    fake_request.session_id = None
    fake_request.department = "General"

    with patch("rlhf.sampler.random.random", return_value=0.999):  # never sample
        with patch("rlhf.generators.api_vs_api.generate_api_vs_api_pair") as mock_gen:
            maybe_collect_pair(fake_request, "response", {})
    mock_gen.assert_not_called()


# ===========================================================================
# Test 6: maybe_collect_pair never raises even on exception
# ===========================================================================

def test_maybe_collect_pair_suppresses_errors():
    from rlhf.sampler import maybe_collect_pair

    fake_request = MagicMock()
    fake_request.prompt = "Test"
    fake_request.session_id = None
    fake_request.department = "HR"

    with patch("rlhf.sampler.random.random", return_value=0.0):
        with patch(
            "rlhf.generators.api_vs_api.generate_api_vs_api_pair",
            side_effect=RuntimeError("API down"),
        ):
            # Must not raise
            maybe_collect_pair(fake_request, "resp", {})


# ===========================================================================
# Test 7: FeedbackEvaluator stores override pair when error type detected
# ===========================================================================

def test_feedback_evaluator_stores_pair_on_override():
    from backend.feedback.evaluator import FeedbackEvaluator

    stored = []

    def fake_write(pair, path=None):
        stored.append(pair)

    def fake_update(pair_id, chosen, labeled_by, judge_metadata=None, path=None):
        pass

    with patch("rlhf.storage.json_store.write_pair", side_effect=fake_write):
        with patch("rlhf.storage.json_store.update_label", side_effect=fake_update):
            with patch("backend.shared.llm_simulator.generate", return_value="safe response"):
                evaluator = FeedbackEvaluator()
                result = evaluator.record_override(
                    request_id="req-001",
                    original_action="ALLOW",
                    final_action="BLOCK",
                    notes="Should have been blocked",
                    prompt="Tell me about salary",
                    original_response="Salary is $85,000",
                )

    assert result["error_type"] == "false_negative"
    assert len(stored) == 1
    assert stored[0].prompt == "Tell me about salary"


# ===========================================================================
# Test 8: FeedbackEvaluator does NOT store a pair when actions match
# ===========================================================================

def test_feedback_evaluator_no_pair_when_actions_match():
    from backend.feedback.evaluator import FeedbackEvaluator

    with patch("rlhf.storage.json_store.write_pair") as mock_write:
        evaluator = FeedbackEvaluator()
        result = evaluator.record_override(
            request_id="req-002",
            original_action="BLOCK",
            final_action="BLOCK",
            notes="",
            prompt="some prompt",
            original_response="response",
        )

    assert result["error_type"] is None
    mock_write.assert_not_called()


# ===========================================================================
# Test 9: /v1/rlhf/status endpoint returns valid structure
# ===========================================================================

def test_rlhf_status_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    resp = client.get("/v1/rlhf/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_pairs" in data
    assert "labeled_pairs" in data
    assert "export_ready" in data
    assert "daily_counts" in data


# ===========================================================================
# Test 10: filter_pairs drops unlabelled and tie pairs
# ===========================================================================

def test_filter_pairs_drops_unlabelled_and_ties():
    from rlhf.export.filters import filter_pairs

    pairs = [
        _make_preference_pair(chosen=None),    # unlabelled => dropped
        _make_preference_pair(chosen="tie"),   # tie => dropped
        _make_preference_pair(chosen="a"),     # OK => kept
        _make_preference_pair(chosen="b"),     # OK => kept
    ]
    result = filter_pairs(pairs)
    assert len(result) == 2
    assert all(p.chosen in ("a", "b") for p in result)


# ===========================================================================
# Test 11: export_for_dpo writes a valid JSONL file with correct columns
# ===========================================================================

def test_export_for_dpo_produces_valid_jsonl(tmp_path):
    from rlhf.export.dpo_export import export_for_dpo
    from rlhf.config import Category
    from rlhf.storage.json_store import write_pair
    from rlhf.schema import ModelResponse, PreferencePair

    # Write a labelled pair to a temp JSONL.
    store_path = tmp_path / "pairs.jsonl"
    pair = PreferencePair(
        prompt="What is the leave policy?",
        response_a=ModelResponse(text="You get 15 days", model_name="m1"),
        response_b=ModelResponse(text="You get 10 days", model_name="m2"),
        category=Category.HR,
        chosen="a",
        labeled_by="human",
    )
    write_pair(pair, path=store_path)

    # export_for_dpo already accepts store_path — no monkey-patching needed.
    out_path = export_for_dpo(
        category=Category.HR,
        output_dir=str(tmp_path),
        store_path=store_path,
    )

    assert Path(out_path).exists()
    lines = Path(out_path).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "prompt" in record
    assert "chosen" in record
    assert "rejected" in record


# ===========================================================================
# Test 12: _infer_category maps departments correctly
# ===========================================================================

def test_infer_category_mapping():
    from rlhf.sampler import _infer_category
    from rlhf.config import Category

    assert _infer_category("HR") == Category.HR
    assert _infer_category("hr") == Category.HR
    assert _infer_category("Finance") == Category.FINANCIAL
    assert _infer_category("FINANCIAL") == Category.FINANCIAL
    assert _infer_category("Engineering") == Category.GENERAL
    assert _infer_category(None) == Category.GENERAL
    assert _infer_category("") == Category.GENERAL

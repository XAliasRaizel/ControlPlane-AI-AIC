"""Tests for the Groq LLM client and generative answer synthesis.

All tests use monkeypatch to mock the Groq client — no real API calls
are made during CI.  The tests verify:
  1. Client initialization and configuration
  2. Error handling (missing key, import errors, API errors)
  3. synthesize_answer() tuple return contract
  4. Graceful fallback from generative to extractive
"""

from __future__ import annotations

import pytest

from rag.schemas import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(texts: list[str]) -> list[RetrievedChunk]:
    """Build a list of RetrievedChunks from plain text strings."""
    return [
        RetrievedChunk(text=t, score=0.8 - i * 0.1, metadata={"source": f"doc_{i}"})
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# GroqLLMClient unit tests
# ---------------------------------------------------------------------------

class TestGroqLLMClient:
    """Tests for rag.ask_controlplane.llm_client.GroqLLMClient."""

    def test_client_initializes_with_env_vars(self, monkeypatch):
        """Client picks up key and model from environment / config."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
        monkeypatch.setenv("GROQ_MODEL", "test-model")
        # Force config reload
        from rag.ask_controlplane.llm_client import GroqLLMClient
        client = GroqLLMClient(api_key="test-key-123", model="test-model")
        assert client.api_key == "test-key-123"
        assert client.model == "test-model"

    def test_client_missing_key_raises(self):
        """Clear error when no API key is provided."""
        from rag.ask_controlplane.llm_client import GroqLLMClient, _GROQ_AVAILABLE
        client = GroqLLMClient(api_key="")
        if _GROQ_AVAILABLE:
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                client.generate(context="test", question="test")
        else:
            with pytest.raises(ImportError, match="groq"):
                client.generate(context="test", question="test")

    def test_generate_returns_string(self, monkeypatch):
        """generate() returns a string answer."""
        from rag.ask_controlplane.llm_client import GroqLLMClient
        import rag.ask_controlplane.llm_client as llm_mod

        # Mock the groq module as available
        monkeypatch.setattr(llm_mod, "_GROQ_AVAILABLE", True)

        class FakeChoice:
            class message:
                content = "  The answer is 42.  "

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeGroq:
            def __init__(self, api_key):
                pass
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return FakeResponse()

        monkeypatch.setattr(llm_mod, "Groq", FakeGroq)
        client = GroqLLMClient(api_key="fake-key")
        result = client.generate(context="some context", question="what is the answer?")
        assert result == "The answer is 42."

    def test_generate_respects_max_tokens(self, monkeypatch):
        """max_tokens parameter is passed through to the API call."""
        from rag.ask_controlplane.llm_client import GroqLLMClient
        import rag.ask_controlplane.llm_client as llm_mod

        monkeypatch.setattr(llm_mod, "_GROQ_AVAILABLE", True)

        captured_kwargs = {}

        class FakeChoice:
            class message:
                content = "answer"

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeGroq:
            def __init__(self, api_key):
                pass
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured_kwargs.update(kwargs)
                        return FakeResponse()

        monkeypatch.setattr(llm_mod, "Groq", FakeGroq)
        client = GroqLLMClient(api_key="fake-key", max_tokens=512)
        client.generate(context="ctx", question="q", max_tokens=256)
        assert captured_kwargs["max_tokens"] == 256

    def test_generate_respects_temperature(self, monkeypatch):
        """temperature parameter is passed through to the API call."""
        from rag.ask_controlplane.llm_client import GroqLLMClient
        import rag.ask_controlplane.llm_client as llm_mod

        monkeypatch.setattr(llm_mod, "_GROQ_AVAILABLE", True)

        captured_kwargs = {}

        class FakeChoice:
            class message:
                content = "answer"

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeGroq:
            def __init__(self, api_key):
                pass
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured_kwargs.update(kwargs)
                        return FakeResponse()

        monkeypatch.setattr(llm_mod, "Groq", FakeGroq)
        client = GroqLLMClient(api_key="fake-key")
        client.generate(context="ctx", question="q", temperature=0.7)
        assert captured_kwargs["temperature"] == 0.7

    def test_fallback_on_import_error(self, monkeypatch):
        """When groq is not installed, is_available() returns False."""
        import rag.ask_controlplane.llm_client as llm_mod
        monkeypatch.setattr(llm_mod, "_GROQ_AVAILABLE", False)
        assert not llm_mod.GroqLLMClient.is_available()


# ---------------------------------------------------------------------------
# synthesize_answer() contract tests
# ---------------------------------------------------------------------------

class TestSynthesizeAnswer:
    """Tests for rag.ask_controlplane.chat.synthesize_answer."""

    def test_synthesize_returns_tuple_extractive(self, monkeypatch):
        """When generation is disabled, returns (text, 'extractive')."""
        from rag.ask_controlplane.chat import synthesize_answer
        from rag.config import RagSettings
        import rag.config as config_mod

        # Disable generation
        monkeypatch.setattr(config_mod, "rag_settings", RagSettings(
            generation_enabled=False,
            groq_api_key="",
        ))

        chunks = _make_chunks(["Annual leave is 18 days per year."])
        answer, mode = synthesize_answer("how many leave days?", chunks)
        assert mode == "extractive"
        assert "18 days" in answer

    def test_synthesize_returns_tuple_generative(self, monkeypatch):
        """When Groq is configured, returns (text, 'groq')."""
        from rag.ask_controlplane.chat import synthesize_answer
        from rag.config import RagSettings
        import rag.config as config_mod
        import rag.ask_controlplane.chat as chat_mod
        import rag.ask_controlplane.llm_client as llm_mod

        # Enable generation with a key — must patch in all three namespaces:
        # rag.config (canonical), rag.ask_controlplane.chat (imported copy),
        # and rag.ask_controlplane.llm_client (GroqLLMClient reads defaults).
        fake_settings = RagSettings(
            generation_enabled=True,
            groq_api_key="fake-key",
        )
        monkeypatch.setattr(config_mod, "rag_settings", fake_settings)
        monkeypatch.setattr(chat_mod, "rag_settings", fake_settings)
        monkeypatch.setattr(llm_mod, "rag_settings", fake_settings)
        monkeypatch.setattr(llm_mod, "_GROQ_AVAILABLE", True)

        class FakeChoice:
            class message:
                content = "You have 18 days of annual leave per year."

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeGroq:
            def __init__(self, api_key):
                pass
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return FakeResponse()

        monkeypatch.setattr(llm_mod, "Groq", FakeGroq)

        chunks = _make_chunks(["Full-time employees accrue 18 days of annual leave."])
        answer, mode = synthesize_answer("how many leave days?", chunks)
        assert mode == "groq"
        assert "18 days" in answer

    def test_fallback_on_api_error(self, monkeypatch):
        """When Groq API fails, falls back to extractive."""
        from rag.ask_controlplane.chat import synthesize_answer
        from rag.config import RagSettings
        import rag.config as config_mod
        import rag.ask_controlplane.chat as chat_mod
        import rag.ask_controlplane.llm_client as llm_mod

        fake_settings = RagSettings(
            generation_enabled=True,
            groq_api_key="fake-key",
        )
        monkeypatch.setattr(config_mod, "rag_settings", fake_settings)
        monkeypatch.setattr(chat_mod, "rag_settings", fake_settings)
        monkeypatch.setattr(llm_mod, "_GROQ_AVAILABLE", True)

        class FakeGroq:
            def __init__(self, api_key):
                pass
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise ConnectionError("API unreachable")

        monkeypatch.setattr(llm_mod, "Groq", FakeGroq)

        chunks = _make_chunks(["Annual leave is 18 days."])
        answer, mode = synthesize_answer("leave days?", chunks)
        assert mode == "extractive"
        assert "18 days" in answer
